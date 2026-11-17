"""Typed job events with a single terminal emitter.

Every long-running operation emits stages on one envelope:
`{seq, ts, event, job_id, shot_id, project_id, kind, stage, progress, ...}`.
The terminal event is written to the database once and emitted once.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import config
from events import event_manager

logger = logging.getLogger("job.events")

TERMINAL_STATUS = {"completed", "failed", "cancelled"}
_meta: dict[str, dict[str, Any]] = {}
_terminal: set[str] = set()
_tasks: set[asyncio.Task] = set()
_lock = threading.RLock()


def register(job_id: str, *, shot_id: str | None = None, project_id: str | None = None, kind: str | None = None) -> None:
    with _lock:
        row = _meta.setdefault(job_id, {})
        if shot_id is not None:
            row["shot_id"] = shot_id
        if project_id is not None:
            row["project_id"] = project_id
        if kind is not None:
            row["kind"] = kind


def _load_meta(job_id: str) -> dict[str, Any]:
    row = _meta.get(job_id) or {}
    if row.get("shot_id") is not None or row.get("project_id") is not None:
        return row
    try:
        from sqlmodel import Session

        from database import Job, engine

        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                row = {
                    "shot_id": job.shot_id,
                    "project_id": job.project_id,
                    "kind": job.type,
                }
                _meta[job_id] = row
    except Exception as exc:
        logger.debug("Could not load job meta for %s: %s", job_id, exc)
    return row


def _payload(
    event: str,
    job_id: str,
    *,
    stage: str | None = None,
    progress: int | None = None,
    step: int | None = None,
    total: int | None = None,
    eta_seconds: int | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = _load_meta(job_id)
    payload: dict[str, Any] = {
        "ts": time.time(),
        "event": event,
        "job_id": job_id,
        "shot_id": meta.get("shot_id"),
        "project_id": meta.get("project_id"),
        "kind": meta.get("kind"),
    }
    if stage is not None:
        payload["stage"] = stage
    if progress is not None:
        payload["progress"] = int(progress)
    if step is not None:
        payload["step"] = int(step)
    if total is not None:
        payload["total"] = int(total)
    if eta_seconds is not None:
        payload["eta_seconds"] = int(eta_seconds)
    if message:
        payload["message"] = message
    if data:
        payload.update(data)
    return payload


async def emit(event: str, job_id: str, **fields: Any) -> None:
    await event_manager.broadcast(event, _payload(event, job_id, **fields))


def emit_threadsafe(event: str, job_id: str, **fields: Any) -> None:
    payload = _payload(event, job_id, **fields)
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(event_manager.broadcast(event, payload))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        return
    except RuntimeError:
        pass
    try:
        import job_utils

        loop = job_utils.global_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(event_manager.broadcast(event, payload), loop)
    except Exception as exc:
        logger.warning("Could not emit %s for %s: %s", event, job_id, exc)


async def queued(job_id: str, *, position: int | None = None, estimate_seconds: float | None = None, message: str = "Queued") -> None:
    data: dict[str, Any] = {}
    if position is not None:
        data["position"] = position
    if estimate_seconds is not None:
        data["estimate_seconds"] = estimate_seconds
    await emit("job.queued", job_id, stage="queued", progress=0, message=message, data=data)


async def waiting(job_id: str, *, reason: str, message: str, position: int | None = None) -> None:
    data = {"reason": reason}
    if position is not None:
        data["position"] = position
    await emit("job.waiting", job_id, stage="waiting", message=message, data=data)


async def started(job_id: str, *, stage: str = "starting", message: str = "Starting…") -> None:
    await emit("job.started", job_id, stage=stage, progress=0, message=message)


async def stage(
    job_id: str,
    stage_name: str,
    *,
    message: str | None = None,
    progress: int | None = None,
    step: int | None = None,
    total: int | None = None,
    eta_seconds: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    await emit(
        "job.stage", job_id,
        stage=stage_name, message=message, progress=progress, step=step,
        total=total, eta_seconds=eta_seconds, data=data,
    )


def progress_threadsafe(
    job_id: str,
    progress: int,
    *,
    stage: str | None = None,
    message: str | None = None,
    step: int | None = None,
    total: int | None = None,
    eta_seconds: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    emit_threadsafe(
        "job.stage" if stage else "job.progress",
        job_id,
        stage=stage, progress=progress, message=message, step=step,
        total=total, eta_seconds=eta_seconds, data=data,
    )


async def retrying(job_id: str, *, attempt: int, message: str, data: dict[str, Any] | None = None) -> None:
    payload = {"attempt": attempt}
    if data:
        payload.update(data)
    await emit("job.retrying", job_id, stage="waiting", message=message, data=payload)


async def cancelling(job_id: str, *, message: str = "Cancelling…") -> None:
    await emit("job.cancelling", job_id, stage="cancelling", message=message)


async def terminal(
    job_id: str,
    status: str,
    *,
    stage: str | None = None,
    message: str | None = None,
    error: str | None = None,
    output_path: str | None = None,
    thumbnail_path: str | None = None,
    actual_frames: int | None = None,
    enhanced_prompt: str | None = None,
    progress: int | None = None,
    data: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
) -> bool:
    """Write once, emit once. Returns False if a terminal event already fired."""
    status = status if status in TERMINAL_STATUS else "failed"
    with _lock:
        if job_id in _terminal:
            return False
        _terminal.add(job_id)

    if memory is None:
        try:
            from services.memory import sampler

            memory = sampler.stop()
        except Exception:
            memory = None
    try:
        from memory_manager import memory_manager

        memory_manager.release_job(job_id)
    except Exception:
        pass

    # Database first, so any reconnect/resync sees the terminal state.
    try:
        import job_utils

        job_utils.update_job_db(
            job_id,
            status,
            output_path=output_path,
            thumbnail_path=thumbnail_path,
            error=error,
            enhanced_prompt=enhanced_prompt,
            status_message=message,
            actual_frames=actual_frames,
            stage=stage,
            progress=progress,
            memory=memory,
        )
    except Exception as exc:
        logger.error("Could not persist terminal state for %s: %s", job_id, exc)

    event = f"job.{status}"
    payload: dict[str, Any] = {}
    if output_path:
        payload["url"] = output_path
    if thumbnail_path:
        payload["thumbnail_url"] = thumbnail_path
    if actual_frames is not None:
        payload["actual_frames"] = actual_frames
    if error:
        payload["error"] = error
    if memory:
        payload["memory"] = memory
    if data:
        payload.update(data)
    await emit(
        event, job_id,
        stage=stage or status,
        progress=progress if progress is not None else (100 if status == "completed" else 0),
        message=message or ("Done" if status == "completed" else error or status.title()),
        data=payload,
    )
    _post_webhook(event, payload)
    with _lock:
        _meta.pop(job_id, None)
    return True


def _post_webhook(event: str, payload: dict[str, Any]) -> None:
    """Best-effort local automation hook; never blocks or fails the job."""
    if not config.JOB_WEBHOOK_URL:
        return
    import json
    import threading
    import urllib.request

    def send() -> None:
        try:
            body = json.dumps({"event": event, "job": payload}, default=str).encode()
            request = urllib.request.Request(
                config.JOB_WEBHOOK_URL,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {config.JOB_WEBHOOK_TOKEN}"}
                        if config.JOB_WEBHOOK_TOKEN else {}
                    ),
                },
            )
            with urllib.request.urlopen(request, timeout=5.0):
                pass
        except Exception as exc:
            logger.warning("Job webhook failed: %s", exc)

    threading.Thread(target=send, name="job-webhook", daemon=True).start()


def reset(job_id: str) -> None:
    """Allow a new terminal event (tests and explicit retries)."""
    with _lock:
        _terminal.discard(job_id)


def terminal_fired(job_id: str) -> bool:
    with _lock:
        return job_id in _terminal
