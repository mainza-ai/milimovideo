import logging
import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlmodel import Session
from database import engine, Job, Shot
from events import event_manager
import config

logger = logging.getLogger(__name__)

# ── Concurrent GPU Job Limiter ─────────────────────────────────────
# Only 1 GPU-intensive job at a time (video gen, image gen, inpainting).
# Additional jobs wait in a FIFO queue via the semaphore.
_gpu_semaphore = None
_gpu_queue_count = 0  # Track how many jobs are waiting

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
    try:
        async with sem:
            acquired = True
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
    """Persist a dispatch record, then start it without blocking the request."""
    from database import QueuedJob
    from sqlmodel import select

    payload = {**(params or {}), **extra}
    with Session(engine) as session:
        existing = session.exec(
            select(QueuedJob)
            .where(QueuedJob.job_id == job_id)
            .where(QueuedJob.status.in_(["queued", "running"]))
        ).first()
        if existing:
            return existing.id
        queued = QueuedJob(
            job_id=job_id,
            kind=kind,
            params_json=json.dumps(payload),
        )
        session.add(queued)
        session.commit()
        session.refresh(queued)
        queue_id = queued.id

    asyncio.create_task(run_queued_job(queue_id))
    return queue_id


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

    with Session(engine) as session:
        queued = session.get(QueuedJob, queue_id)
        if not queued or queued.status != "queued":
            return
        queued.status = "running"
        queued.attempts = (queued.attempts or 0) + 1
        from datetime import datetime
        queued.started_at = datetime.utcnow()
        params = json.loads(queued.params_json or "{}")
        job_id = queued.job_id
        kind = queued.kind
        session.add(queued)
        session.commit()

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
        else:
            raise ValueError(f"Unknown queued job kind: {kind}")
        status = "completed"
        error = None
    except Exception as exc:
        from database import QueuedJob

        attempts = 0
        with Session(engine) as session:
            current = session.get(QueuedJob, queue_id)
            attempts = (current.attempts or 0) if current else 0

        if _is_oom(exc) and attempts < 2 and kind in ("video", "image"):
            retry_params = _oom_retry_params(kind, params)
            logger.warning(
                "OOM in job %s; retrying once at %sx%s / %s frames",
                job_id,
                retry_params.get("width"),
                retry_params.get("height"),
                retry_params.get("num_frames"),
            )
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
            update_job_db(job_id, "pending", status_message="OOM: retrying at a smaller preset")
            shot_id = retry_params.get("shot_id") or retry_params.get("id")
            if shot_id:
                update_shot_db(
                    shot_id,
                    width=retry_params.get("width"),
                    height=retry_params.get("height"),
                    num_frames=retry_params.get("num_frames"),
                )
            await broadcast_log(
                job_id,
                "Out of memory: retrying once at "
                f"{retry_params.get('width')}x{retry_params.get('height')}, "
                f"{retry_params.get('num_frames')} frames",
            )
            await run_queued_job(queue_id)
            return

        logger.exception(f"Queued job {job_id} failed: {exc}")
        status = "failed"
        error = str(exc)
        update_job_db(job_id, "failed", error=error)

    with Session(engine) as session:
        current = session.get(QueuedJob, queue_id)
        if current:
            from datetime import datetime
            current.status = status
            current.error_message = error
            current.completed_at = datetime.utcnow()
            session.add(current)
            session.commit()


async def resume_queued_jobs() -> None:
    """Recover dispatch records after a restart.

    Running jobs get one retry; queued jobs start immediately. The GPU semaphore
    still serializes them, so recovery cannot load two heavy models at once.
    """
    from database import QueuedJob
    from sqlmodel import select

    with Session(engine) as session:
        running = session.exec(
            select(QueuedJob).where(QueuedJob.status == "running")
        ).all()
        for queued in running:
            if (queued.attempts or 0) < 2:
                queued.status = "queued"
            else:
                queued.status = "failed"
                queued.error_message = "Interrupted twice; not retrying"
            session.add(queued)
        session.commit()

        pending = session.exec(
            select(QueuedJob)
            .where(QueuedJob.status == "queued")
            .order_by(QueuedJob.created_at)
        ).all()
        ids = [q.id for q in pending]

    for queue_id in ids:
        asyncio.create_task(run_queued_job(queue_id))

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

def update_job_progress(job_id: str, progress: int, message: str = None, **kwargs):
    """Updates in-memory state and broadcasts progress."""
    if job_id in active_jobs:
        active_jobs[job_id]["progress"] = progress
        if message:
            active_jobs[job_id]["status_message"] = message
    
    # Thread-safe broadcast
    import asyncio
    global global_loop
    
    try:
        # First try getting the running loop (works if called from async function)
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_progress(job_id, progress, "processing", message, **kwargs))
    except RuntimeError:
        # We are likely in a thread (ThreadPoolExecutor)
        # Use the captured global loop
        if global_loop and global_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(job_id, progress, "processing", message, **kwargs),
                global_loop
            )
        else:
            logger.warning(f"Could not broadcast progress for {job_id}: No event loop available")
    except Exception as e:
        logger.error(f"Broadcast failed: {e}")


def update_job_db(job_id: str, status: str, output_path: str = None, error: str = None, enhanced_prompt: str = None, status_message: str = None, actual_frames: int = None, thumbnail_path: str = None):
    """Helper to update Job record in SQLite."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = status
                # Force JSON string update if needed, but usually SQLModel handles explicit fields.
                # job.params_json = job.params_json 
                if output_path: 
                    job.output_path = output_path
                if error: 
                    job.error_message = str(error)
                if enhanced_prompt:
                    job.enhanced_prompt = enhanced_prompt
                if status_message:
                    job.status_message = status_message
                if actual_frames is not None:
                    job.actual_frames = actual_frames
                if thumbnail_path:
                    job.thumbnail_path = thumbnail_path
                if status in ["completed", "failed", "cancelled"]:
                    job.completed_at = datetime.now(timezone.utc)
                    # CLEANUP: Remove from active_jobs so status endpoint falls back to DB
                    if job_id in active_jobs:
                        del active_jobs[job_id]
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
    data = {"job_id": job_id, "progress": progress, "status": status}
    if message:
        data["message"] = message
    data.update(kwargs)
    await event_manager.broadcast("progress", data)

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
