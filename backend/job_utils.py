import logging
import os
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
_gpu_semaphore = asyncio.Semaphore(1)
_gpu_queue_count = 0  # Track how many jobs are waiting

async def gpu_job_wrapper(coro):
    """Wraps a GPU task coroutine so only one runs at a time."""
    global _gpu_queue_count
    _gpu_queue_count += 1
    logger.info(f"GPU job queued ({_gpu_queue_count} in queue)")
    
    try:
        async with _gpu_semaphore:
            _gpu_queue_count -= 1
            logger.info(f"GPU job acquired lock ({_gpu_queue_count} waiting)")
            return await coro
    except Exception as e:
        _gpu_queue_count = max(0, _gpu_queue_count - 1)
        raise

def is_gpu_busy() -> dict:
    """Returns GPU busy status for the frontend."""
    return {
        "busy": _gpu_semaphore.locked(),
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
