import logging
import os
import json
import asyncio
import time
from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlmodel import Session
from database import engine, Job, Shot
from events import event_manager
from services.memory.models import MemoryDeferred
import config

logger = logging.getLogger(__name__)

# ── Concurrent GPU Job Limiter ─────────────────────────────────────
# Only 1 GPU-intensive job at a time (video gen, image gen, inpainting).
# Additional jobs wait in a FIFO queue via the semaphore.
_gpu_semaphore = None
_gpu_queue_count = 0  # Track how many jobs are waiting
_queue_paused = False
_background_tasks: set[asyncio.Task] = set()
_queue_waits: deque[float] = deque(maxlen=200)


def set_queue_paused(paused: bool) -> bool:
    """Pause/resume dispatch of queued GPU jobs without killing running work."""
    global _queue_paused
    _queue_paused = bool(paused)
    logger.info("GPU queue %s", "paused" if _queue_paused else "resumed")
    return _queue_paused


def is_queue_paused() -> bool:
    return _queue_paused


def _spawn_background(coro, loop=None) -> asyncio.Task:
    """Create a task and keep it alive until it reaches a terminal state."""
    task = loop.create_task(coro) if loop else asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

def get_gpu_semaphore():
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(1)
    return _gpu_semaphore

async def gpu_job_wrapper(coro):
    """Wraps a GPU task coroutine so only one runs at a time."""
    global _gpu_queue_count
    _gpu_queue_count += 1
    logger.info(f"GPU job queued ({_gpu_queue_count} in queue)")
    
    sem = get_gpu_semaphore()
    acquired = False
    waiting_since = time.monotonic()
    try:
        async with sem:
            acquired = True
            _queue_waits.append(time.monotonic() - waiting_since)
            _gpu_queue_count = max(0, _gpu_queue_count - 1)
            logger.info(f"GPU job acquired lock ({_gpu_queue_count} waiting)")
            return await coro
    except Exception:
        if not acquired:
            _gpu_queue_count = max(0, _gpu_queue_count - 1)
        raise

async def queue_video_task(job_id: str, params: dict):
    """Helper to wrap video generation in the queue"""
    from tasks.video import generate_video_task
    await gpu_job_wrapper(generate_video_task(job_id, params))


async def queue_image_task(job_id: str, params: dict):
    """All Flux/thumbnail work must share the same GPU slot as video."""
    from tasks.image import generate_image_task
    await gpu_job_wrapper(generate_image_task(job_id, params))


async def queue_inpaint_task(job_id: str, image_path: str, mask_path: str, prompt: str):
    from managers.inpainting_manager import inpainting_manager
    await gpu_job_wrapper(
        inpainting_manager.process_inpaint(job_id, image_path, mask_path, prompt)
    )


async def queue_ai_parse_task(job_id: str, params: dict):
    from tasks.ai_parse import ai_parse_task
    await gpu_job_wrapper(ai_parse_task(job_id, params))


async def queue_depth_task(job_id: str, params: dict):
    from tasks.depth import depth_task
    await gpu_job_wrapper(depth_task(job_id, params))


async def queue_element_visual_task(job_id: str, element_id: str, prompt_override=None,
                                    guidance: float = 2.0, enable_ae: bool = False):
    from managers.element_manager import element_manager
    await gpu_job_wrapper(
        element_manager.generate_visual_task(
            job_id, element_id, prompt_override, guidance, enable_ae
        )
    )


# ── Durable dispatch ledger ───────────────────────────────────────

async def enqueue_gpu_job(job_id: str, kind: str, params: dict, **extra) -> str:
    """Persist a dispatch record, announce it, then start it without blocking."""
    from database import QueuedJob
    from sqlmodel import select

    import job_events

    payload = {**(params or {}), **extra}
    with Session(engine) as session:
        existing = session.exec(
            select(QueuedJob)
            .where(QueuedJob.job_id == job_id)
            .where(QueuedJob.status.in_(["queued", "running"]))
        ).first()
        if existing:
            return existing.id
        ahead = session.exec(
            select(QueuedJob).where(QueuedJob.status.in_(["queued", "running"]))
        ).all()
        position = len(ahead) + 1
        queued = QueuedJob(
            job_id=job_id,
            kind=kind,
            params_json=json.dumps(payload),
        )
        session.add(queued)
        session.commit()
        session.refresh(queued)
        queue_id = queued.id

        job = session.get(Job, job_id)
        if job:
            job.status = "queued"
            job.status_message = "Queued"
            job.stage = "queued"
            session.add(job)
            session.commit()
        shot_id = (job.shot_id if job else None) or payload.get("shot_id")
        project_id = (job.project_id if job else None) or payload.get("project_id")

    job_events.register(job_id, shot_id=shot_id, project_id=project_id, kind=kind)
    frames = int(payload.get("num_frames") or 121)
    width = int(payload.get("width") or 768)
    height = int(payload.get("height") or 512)
    workload = max(0.1, (frames / 121.0) * ((width * height) / (768 * 512)))
    estimate_seconds = round(8 + 22 * workload, 1)
    await job_events.queued(
        job_id,
        position=position,
        estimate_seconds=estimate_seconds,
        message=f"Queued (#{position})",
    )
    _spawn_background(run_queued_job(queue_id))
    return queue_id


# Interactive work outranks bulk thumbnails/elements, which outrank background
# parsing. Within a tier the queue stays FIFO.
KIND_PRIORITY = {
    "video": 20,
    "image": 20,
    "inpaint": 30,
    "element": 50,
    "ai_parse": 60,
    "depth": 60,
}


def job_priority(kind: str) -> int:
    return KIND_PRIORITY.get(kind, 50)


def job_slot(kind: str) -> str:
    """The residency slot a job will load, for same-slot grouping."""
    if kind in ("image", "inpaint", "element"):
        return "image"
    if kind == "depth":
        return "depth"
    if kind in ("video", "ai_parse"):
        return "video"
    return "other"


def _row_params(row) -> dict:
    try:
        return json.loads(row.params_json or "{}")
    except Exception:
        return {}


def row_priority(row) -> int:
    """Explicit per-job priority (0 highest, 100 lowest), else the kind default."""
    try:
        value = _row_params(row).get("priority")
        if value is not None:
            return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        pass
    return job_priority(row.kind)


def row_not_before(row) -> float:
    try:
        return float(_row_params(row).get("_memory_not_before") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _queue_head(session, active_slots: set[str] | None = None):
    """Next dispatch candidate: priority, then same-slot grouping, then FIFO.

    Rows sleeping after a memory refusal are skipped until their retry time.
    """
    from database import QueuedJob
    from sqlmodel import select

    # Only queued rows contend for a turn. A running row has already passed it.
    rows = session.exec(
        select(QueuedJob).where(QueuedJob.status == "queued")
    ).all()
    now = time.time()
    rows = [row for row in rows if row_not_before(row) <= now]
    if not rows:
        return None
    active = active_slots or set()
    return min(
        rows,
        key=lambda row: (
            row_priority(row),
            0 if job_slot(row.kind) in active else 1,
            row.created_at,
            row.id,
        ),
    )


# A live pump marks its row running within one poll, so a queued head that
# outlives this window has no worker (crash, failed spawn, seed row). Proceeding
# bounds starvation instead of blocking the whole queue forever.
MAX_TURN_WAIT_SEC = 3.0


async def wait_for_turn(queue_id: str) -> bool:
    """Block until this row is the highest-priority dispatch candidate.

    The GPU semaphore is first-come, so ordering has to happen before the lock.
    Returns False if the row was cancelled while waiting.
    """
    from database import QueuedJob

    from memory_manager import memory_manager

    deadline = time.monotonic() + MAX_TURN_WAIT_SEC
    while True:
        with Session(engine) as session:
            row = session.get(QueuedJob, queue_id)
            if not row or row.status != "queued":
                return False
            if row_not_before(row) > time.time():
                # A delayed retry owns its own timer; do not spin a pump.
                return False
            head = _queue_head(session, active_slots=set(memory_manager.active))
            if head and head.id == queue_id:
                return True
            if not head:
                return True
            if time.monotonic() >= deadline:
                logger.warning(
                    "Queue turn wait timed out for %s behind %s; dispatching anyway",
                    queue_id, head.job_id if head else "none",
                )
                return True
        await asyncio.sleep(0.25)


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return any(token in text for token in (
        "out of memory",
        "cuda oom",
        "mps backend out of memory",
        "allocation failed",
        "not enough memory",
    ))


def _oom_retry_params(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    retry = dict(params)
    if "width" in retry:
        retry["width"] = max(256, int(round(int(retry["width"]) * 0.75 / 8) * 8))
    if "height" in retry:
        retry["height"] = max(256, int(round(int(retry["height"]) * 0.75 / 8) * 8))
    if kind == "video" and "num_frames" in retry:
        reduced = max(49, int(round(int(retry["num_frames"]) * 0.75)))
        retry["num_frames"] = ((reduced - 1) // 8) * 8 + 1
        # Upscaling is the memory peak; drop it on the retry.
        retry["upscale"] = False
    if "num_inference_steps" in retry:
        retry["num_inference_steps"] = max(8, int(int(retry["num_inference_steps"]) * 0.75))
    return retry


def _clear_accelerator_cache() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        logger.debug("Could not clear accelerator cache", exc_info=True)


async def run_queued_job(queue_id: str) -> None:
    from database import QueuedJob
    from sqlmodel import select

    import job_events
    from memory_manager import memory_manager

    waiting_emitted = False
    job_id = ""
    kind = "unknown"
    while is_queue_paused():
        with Session(engine) as session:
            queued = session.get(QueuedJob, queue_id)
            if not queued:
                return
            job_id = queued.job_id
            kind = queued.kind
        if not waiting_emitted:
            await job_events.waiting(
                job_id, reason="queue", message="GPU queue is paused; waiting for resume"
            )
            waiting_emitted = True
        await asyncio.sleep(0.25)

    # Priority arbitration happens before the GPU semaphore is taken.
    if not await wait_for_turn(queue_id):
        return

    with Session(engine) as session:
        queued = session.get(QueuedJob, queue_id)
        if not queued or queued.status != "queued":
            return
        job = session.get(Job, queued.job_id)
        job_id = queued.job_id
        kind = queued.kind
        if job and job.status in ("cancelled", "failed"):
            queued.status = "cancelled" if job.status == "cancelled" else "failed"
            queued.error_message = job.error_message or f"Job was {job.status} before dispatch"
            from datetime import datetime
            queued.completed_at = datetime.utcnow()
            session.add(queued)
            session.commit()
            await job_events.terminal(
                job_id,
                job.status,
                stage="cancelled" if job.status == "cancelled" else "failed",
                message=job.error_message or f"Job was {job.status} before dispatch",
                error=job.error_message if job.status == "failed" else None,
            )
            return

        queued.status = "running"
        queued.attempts = (queued.attempts or 0) + 1
        from datetime import datetime
        queued.started_at = datetime.utcnow()
        params = json.loads(queued.params_json or "{}")
        attempts = queued.attempts or 1
        job_shot_id = job.shot_id if job else params.get("shot_id")
        job_project_id = job.project_id if job else params.get("project_id")
        session.add(queued)
        if job:
            job.status = "processing"
            job.started_at = job.started_at or datetime.utcnow()
            job.stage = "starting"
            job.status_message = "Starting…"
            session.add(job)
        session.commit()

    job_events.register(
        job_id,
        shot_id=job_shot_id,
        project_id=job_project_id,
        kind=kind,
    )
    await job_events.started(job_id, message=f"Starting {kind} job…")
    try:
        from services.memory import sampler

        sampler.start(job_id)
    except Exception:
        pass

    status = "completed"
    error = None
    try:
        if kind == "video":
            await queue_video_task(job_id, params)
        elif kind == "image":
            await queue_image_task(job_id, params)
        elif kind == "inpaint":
            await queue_inpaint_task(
                job_id,
                params.get("image_path"),
                params.get("mask_path"),
                params.get("prompt", ""),
            )
        elif kind == "element":
            await queue_element_visual_task(
                job_id,
                params.get("element_id"),
                params.get("prompt_override"),
                params.get("guidance", 2.0),
                params.get("enable_ae", False),
            )
        elif kind == "ai_parse":
            await queue_ai_parse_task(job_id, params)
        elif kind == "depth":
            await queue_depth_task(job_id, params)
        else:
            raise ValueError(f"Unknown queued job kind: {kind}")
    except MemoryDeferred as deferred:
        # Transient refusal: sleep, announce the wait, and re-dispatch without
        # holding the GPU semaphore or a queue turn.
        memory_manager.release_job(job_id)
        try:
            from services.memory import sampler

            sampler.stop()
        except Exception:
            pass
        wait_count = int(params.get("_memory_waits") or 0) + 1
        if wait_count > MAX_MEMORY_WAITS:
            status = "failed"
            error = str(deferred)
            await job_events.terminal(
                job_id, "failed", stage="admission", error=error,
                message=f"Still cannot fit after {MAX_MEMORY_WAITS} waits: {error}",
            )
        else:
            delay = min(10.0 * wait_count, 120.0)
            params["_memory_waits"] = wait_count
            params["_memory_not_before"] = time.time() + delay
            with Session(engine) as session:
                current = session.get(QueuedJob, queue_id)
                if current:
                    current.status = "queued"
                    current.params_json = json.dumps(params)
                    current.started_at = None
                    # A memory wait is not a generation attempt.
                    current.attempts = max(0, (current.attempts or 1) - 1)
                    session.add(current)
                    session.commit()
            update_job_db(
                job_id, "pending",
                status_message=f"Waiting for memory: {deferred}",
                stage="waiting",
            )
            await job_events.waiting(
                job_id,
                reason="memory",
                message=f"Waiting for memory ({wait_count}/{MAX_MEMORY_WAITS}): {deferred}",
            )
            _spawn_background(_delayed_redispatch(queue_id, delay))
            return

    except Exception as exc:
        memory_manager.release_job(job_id)
        if _is_oom(exc) and attempts < 2 and kind in ("video", "image"):
            retry_params = _oom_retry_params(kind, params)
            decision = memory_manager.admit_job(job_id, kind, retry_params)
            if not decision.ok:
                status = "failed"
                error = decision.reason or str(exc)
                await job_events.terminal(
                    job_id, "failed", stage="admission", error=error,
                    message=f"OOM retry refused: {error}",
                )
            else:
                _clear_accelerator_cache()
                with Session(engine) as session:
                    current = session.get(QueuedJob, queue_id)
                    if current:
                        current.status = "queued"
                        current.params_json = json.dumps(retry_params)
                        current.error_message = f"OOM retry: {exc}"[:1000]
                        current.started_at = None
                        session.add(current)
                        session.commit()
                update_job_db(
                    job_id, "pending",
                    status_message="Out of memory: retrying at a smaller preset",
                    stage="waiting",
                )
                shot_id = retry_params.get("shot_id") or retry_params.get("id")
                if shot_id:
                    update_shot_db(
                        shot_id,
                        width=retry_params.get("width"),
                        height=retry_params.get("height"),
                        num_frames=retry_params.get("num_frames"),
                    )
                await job_events.retrying(
                    job_id,
                    attempt=attempts,
                    message=(
                        "Out of memory: retrying once at "
                        f"{retry_params.get('width')}x{retry_params.get('height')}, "
                        f"{retry_params.get('num_frames')} frames, no upscale"
                    ),
                    data={"width": retry_params.get("width"), "height": retry_params.get("height"),
                          "num_frames": retry_params.get("num_frames"), "upscale": False},
                )
                memory_manager.release_job(job_id)
                await run_queued_job(queue_id)
                return
        else:
            logger.exception("Queued job %s failed: %s", job_id, exc)
            status = "failed"
            error = str(exc)
            await job_events.terminal(
                job_id, "failed", stage="failed", error=error,
                message=f"Generation failed: {error}",
            )
    finally:
        memory_manager.release_job(job_id)

    with Session(engine) as session:
        current = session.get(QueuedJob, queue_id)
        if current:
            from datetime import datetime
            current.status = status
            current.error_message = error
            current.completed_at = datetime.utcnow()
            session.add(current)
            session.commit()

    if status == "completed" and not job_events.terminal_fired(job_id):
        await job_events.terminal(job_id, "completed", stage="completed", message="Done")


MAX_MEMORY_WAITS = 20


async def _delayed_redispatch(queue_id: str, delay: float) -> None:
    await asyncio.sleep(max(0.0, delay))
    await run_queued_job(queue_id)


def apply_downgrade(job_id: str, params: dict, downgrade: dict | None) -> dict:
    """Persist a smaller preset to the Job and its dispatch row."""
    if not downgrade:
        return params
    changes = downgrade.get("params") or {}
    if not changes:
        return params
    for key, value in changes.items():
        params[key] = value
    try:
        from database import QueuedJob
        from sqlmodel import select

        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                stored = json.loads(job.params_json or "{}")
                stored.update(params)
                job.params_json = json.dumps(stored)
                session.add(job)
            queued = session.exec(
                select(QueuedJob)
                .where(QueuedJob.job_id == job_id)
                .where(QueuedJob.status.in_(["queued", "running"]))
            ).first()
            if queued:
                stored = json.loads(queued.params_json or "{}")
                stored.update(changes)
                queued.params_json = json.dumps(stored)
                session.add(queued)
            session.commit()
    except Exception as exc:
        logger.warning("Could not persist downgrade for %s: %s", job_id, exc)
    return params


async def resume_queued_jobs() -> None:
    """Reconcile interrupted dispatch records after a restart.

    A job whose worker died is failed, not silently re-run: the memory budget
    must re-admit it first, and the machine may still be under pressure. Queued
    jobs with a live Job row start normally.
    """
    import job_events
    from database import QueuedJob
    from sqlmodel import select

    interrupted: list[tuple[str, str | None, str | None, str]] = []
    dispatch: list[str] = []
    with Session(engine) as session:
        running = session.exec(
            select(QueuedJob).where(QueuedJob.status == "running")
        ).all()
        for queued in running:
            job = session.get(Job, queued.job_id)
            if job and job.status == "completed":
                queued.status = "completed"
                queued.completed_at = job.completed_at or datetime.utcnow()
                session.add(queued)
                continue
            message = "Server restarted while the job was running"
            queued.status = "failed"
            queued.error_message = message
            queued.completed_at = datetime.utcnow()
            session.add(queued)
            if job and job.status not in ("failed", "cancelled"):
                job.status = "failed"
                job.error_message = message
                job.stage = "recovery"
                job.completed_at = datetime.utcnow()
                session.add(job)
            interrupted.append((
                queued.job_id,
                job.shot_id if job else None,
                job.project_id if job else None,
                job.type if job else queued.kind,
            ))

        pending = session.exec(
            select(QueuedJob)
            .where(QueuedJob.status == "queued")
            .order_by(QueuedJob.created_at)
        ).all()
        for queued in pending:
            job = session.get(Job, queued.job_id)
            if not job or job.status in ("failed", "cancelled"):
                queued.status = "failed"
                queued.error_message = "Job was already terminal before dispatch"
                queued.completed_at = datetime.utcnow()
                session.add(queued)
                continue
            dispatch.append(queued.id)
        session.commit()

    for job_id, shot_id, project_id, kind in interrupted:
        job_events.register(job_id, shot_id=shot_id, project_id=project_id, kind=kind)
        await job_events.terminal(
            job_id, "failed", stage="recovery",
            error="Server restarted while the job was running",
            message="Server restarted before the job finished; it was not retried",
        )

    for queue_id in dispatch:
        _spawn_background(run_queued_job(queue_id))


def queue_wait_stats() -> dict:
    waits = sorted(_queue_waits)
    if not waits:
        return {"samples": 0, "wait_p50_s": None, "wait_p95_s": None, "wait_max_s": None}

    def pct(p: float) -> float:
        index = min(len(waits) - 1, max(0, int(round(p * (len(waits) - 1)))))
        return round(waits[index], 2)

    return {
        "samples": len(waits),
        "wait_p50_s": pct(0.50),
        "wait_p95_s": pct(0.95),
        "wait_max_s": round(waits[-1], 2),
    }


def is_gpu_busy() -> dict:
    """Returns GPU busy status for the frontend."""
    return {
        "busy": _gpu_semaphore.locked() if _gpu_semaphore else False,
        "queued": _gpu_queue_count
    }

# Cancellation tracking
# job_id -> {'cancelled': bool}
active_jobs: Dict[str, Dict[str, Any]] = {}

def cancel_all_jobs() -> int:
    """Marks all active jobs as cancelled to stop background threads."""
    count = 0
    # Copy keys to avoid size change issues during iteration
    for job_id in list(active_jobs.keys()):
        if not active_jobs[job_id].get("cancelled"):
            active_jobs[job_id]["cancelled"] = True
            active_jobs[job_id]["status"] = "cancelling"
            logger.warning(f"Shutdown: Flagged job {job_id} for cancellation")
            count += 1
    return count

# Global loop reference for thread-safe broadcasting
global_loop: Optional[Any] = None

_last_progress_persist: dict[str, tuple[float, str | None]] = {}


def _persist_job_progress(
    job_id: str,
    progress: int,
    message: str | None,
    stage: str | None,
    eta_seconds: int | None = None,
) -> None:
    """Persist progress at most once per second (or on a stage change)."""
    now = time.monotonic()
    previous = _last_progress_persist.get(job_id)
    if previous and previous[1] == stage and now - previous[0] < 1.0:
        return
    _last_progress_persist[job_id] = (now, stage)
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.progress = int(progress)
                if message:
                    job.status_message = message
                if stage:
                    job.stage = stage
                if not job.started_at:
                    job.started_at = datetime.utcnow()
                session.add(job)
                session.commit()
    except Exception as exc:
        logger.debug("Could not persist progress for %s: %s", job_id, exc)


def update_job_progress(
    job_id: str,
    progress: int,
    message: str = None,
    *,
    stage: str | None = None,
    step: int | None = None,
    total: int | None = None,
    eta_seconds: int | None = None,
    **kwargs,
):
    """Update memory state, persist (throttled), and emit a typed job event."""
    import job_events
    from services.memory import sampler

    meta = active_jobs.get(job_id)
    if stage and meta is not None and meta.get("stage") != stage:
        # ETA is per stage: upscaling restarts its own step counter.
        meta["stage_started_ts"] = time.time()
    if stage and meta is not None and meta.get("stage_started_ts") is None:
        meta["stage_started_ts"] = time.time()
    if eta_seconds is None and step and total and 0 < step < total and meta is not None:
        started = meta.get("stage_started_ts")
        if started:
            per_step = max(0.001, (time.time() - started) / step)
            eta_seconds = int(per_step * (total - step))
    if job_id in active_jobs:
        active_jobs[job_id]["progress"] = progress
        if message:
            active_jobs[job_id]["status_message"] = message
        if stage:
            active_jobs[job_id]["stage"] = stage
        if step is not None:
            active_jobs[job_id]["step"] = step
        if total is not None:
            active_jobs[job_id]["total"] = total
        if eta_seconds is not None:
            active_jobs[job_id]["eta_seconds"] = eta_seconds
        if kwargs.get("enhanced_prompt"):
            active_jobs[job_id]["enhanced_prompt"] = kwargs["enhanced_prompt"]
    if stage:
        sampler.mark(stage)
    _persist_job_progress(job_id, progress, message, stage, eta_seconds)
    job_events.progress_threadsafe(
        job_id, progress,
        stage=stage, message=message, step=step, total=total,
        eta_seconds=eta_seconds, data=kwargs or None,
    )


def update_job_db(
    job_id: str,
    status: str,
    output_path: str = None,
    error: str = None,
    enhanced_prompt: str = None,
    status_message: str = None,
    actual_frames: int = None,
    thumbnail_path: str = None,
    stage: str = None,
    progress: int | None = None,
    memory: dict | None = None,
    started_at: datetime | None = None,
):
    """Helper to update Job record in SQLite (terminal-aware)."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = status
                if output_path:
                    job.output_path = output_path
                if error is not None:
                    job.error_message = str(error) if error else None
                if enhanced_prompt:
                    job.enhanced_prompt = enhanced_prompt
                    try:
                        from llm import last_enhancement
                        provenance = last_enhancement()
                        if provenance:
                            params = json.loads(job.params_json or "{}")
                            params["llm"] = provenance
                            job.params_json = json.dumps(params)
                    except Exception:
                        pass
                if status_message:
                    job.status_message = status_message
                if actual_frames is not None:
                    job.actual_frames = actual_frames
                if thumbnail_path:
                    job.thumbnail_path = thumbnail_path
                if stage:
                    job.stage = stage
                if progress is not None:
                    job.progress = int(progress)
                elif status in ("completed", "failed", "cancelled"):
                    job.progress = 100 if status == "completed" else job.progress
                if memory:
                    job.memory_json = json.dumps(memory)
                if started_at:
                    job.started_at = started_at
                if status == "processing" and not job.started_at:
                    job.started_at = datetime.utcnow()
                if status in ("completed", "failed", "cancelled"):
                    job.completed_at = datetime.now(timezone.utc)
                    if job_id in active_jobs:
                        del active_jobs[job_id]
                    _last_progress_persist.pop(job_id, None)
                session.add(job)
                session.commit()
    except Exception as e:
        logger.error(f"Failed to update job DB for {job_id}: {e}")

def update_shot_db(shot_id: str, **updates):
    """Helper to update Shot record in SQLite with any provided fields."""
    if not shot_id:
        return
    try:
        with Session(engine) as session:
            shot = session.get(Shot, shot_id)
            if shot:
                for key, value in updates.items():
                    if hasattr(shot, key) and value is not None:
                        setattr(shot, key, value)
                session.add(shot)
                session.commit()
                # logger.info(f"Updated Shot {shot_id}: {list(updates.keys())}")
    except Exception as e:
        logger.error(f"Failed to update shot DB for {shot_id}: {e}")

async def broadcast_log(job_id: str, message: str):
    logger.info(f"[{job_id}] {message}")
    await event_manager.broadcast("log", {"job_id": job_id, "message": message})
    
async def broadcast_progress(job_id: str, progress: int, status: str = "processing", message: str = None, **kwargs):
    """Compatibility helper: emit a typed progress event (no persistence)."""
    import job_events

    stage = kwargs.pop("stage", None)
    await job_events.emit(
        "job.stage" if stage else "job.progress",
        job_id,
        stage=stage,
        progress=progress,
        message=message,
        data=kwargs or None,
    )


def resolve_element_image_path(image_path: str) -> str | None:
    """Resolve element image_path (web URL format) to absolute filesystem path.
    
    Element visuals are stored as web URLs (e.g., /projects/{id}/assets/elements/...)
    but we need absolute paths for image loading.
    """
    if not image_path:
        return None
    
    # Already an absolute path that exists
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # Web URL format: /projects/{id}/assets/...
    if image_path.startswith("/projects/"):
        relative = image_path[len("/projects/"):]  # Safe prefix removal
        full_path = os.path.join(config.PROJECTS_DIR, relative)
        if os.path.exists(full_path):
            return full_path
        else:
            logger.warning(f"Element image not found at resolved path: {full_path}")
    
    return None
