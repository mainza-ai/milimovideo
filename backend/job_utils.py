import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlmodel import Session
from database import engine, Job, Shot
from events import event_manager
import config

logger = logging.getLogger(__name__)

# Cancellation tracking
# job_id -> {'cancelled': bool}
active_jobs: Dict[str, Dict[str, Any]] = {}

def update_job_progress(job_id: str, progress: int, message: str = None):
    """Updates in-memory state and broadcasts progress."""
    if job_id in active_jobs:
        active_jobs[job_id]["progress"] = progress
        if message:
            active_jobs[job_id]["status_message"] = message
    
    # Fire and forget async broadcast (requires running loop or handling)
    # Since this is called from sync code in callbacks, we might need a helper.
    # But event_manager.broadcast is async. 
    # Calling code usually handles the loop (e.g. video.py calls this from sync callback).
    # We can't await properly here if not in async context.
    # BUT, the listener expects broadcast.
    # We will rely on the caller to handle broadcast if they can, OR we rely on a poller.
    # Original worker.py had `run_coroutine_threadsafe` inside the callback.
    # We will let the caller handle broadcast invocation if they are async, 
    # but for sync callbacks, we might need to skip broadcast here and rely on memory + polling?
    # NO: The UI needs events.
    # We'll import asyncio and try to schedule it if loop exists.
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_progress(job_id, progress, "processing"))
    except:
        pass # No loop, can't broadcast


def update_job_db(job_id: str, status: str, output_path: str = None, error: str = None, enhanced_prompt: str = None, status_message: str = None, actual_frames: int = None, thumbnail_path: str = None):
    """Helper to update Job record in SQLite."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = status
                job.params_json = job.params_json
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
    
async def broadcast_progress(job_id: str, progress: int, status: str = "processing"):
    await event_manager.broadcast("progress", {"job_id": job_id, "progress": progress, "status": status})

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
