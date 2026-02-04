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
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_progress(job_id, progress, "processing", message))
    except:
        pass # No loop, can't broadcast


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
    
async def broadcast_progress(job_id: str, progress: int, status: str = "processing", message: str = None):
    data = {"job_id": job_id, "progress": progress, "status": status}
    if message:
        data["message"] = message
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
