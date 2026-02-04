import os
import shutil
import logging
import subprocess
from typing import Optional
import config

# Use centralized config for base directory to ensure consistency
# This replaces the specific __file__ based logic in worker.py and server.py
def get_base_dir():
    """Get backend base directory."""
    return config.BACKEND_DIR

def get_project_output_paths(job_id: str, project_id: str):
    """
    Get output paths for a generation job.
    project_id is REQUIRED - all outputs are stored in project workspaces.
    """
    if not project_id:
        raise ValueError(f"project_id is required for job {job_id}. Legacy paths are no longer supported.")
    
    # Use config.PROJECTS_DIR which is absolute
    projects_dir = os.path.join(config.PROJECTS_DIR, project_id)
    
    return {
        "output_dir": os.path.join(projects_dir, "generated"),
        "thumbnail_dir": os.path.join(projects_dir, "thumbnails"),
        "workspace_dir": os.path.join(projects_dir, "workspace"),
        "output_path": os.path.join(projects_dir, "generated", f"{job_id}.mp4"),
        "thumbnail_path": os.path.join(projects_dir, "thumbnails", f"{job_id}_thumb.jpg"),
        "project_id": project_id
    }

def generate_thumbnail(video_path: str) -> Optional[str]:
    thumb_path = os.path.splitext(video_path)[0] + "_thumb.jpg"
    # If already exists, return
    if os.path.exists(thumb_path):
        return thumb_path
        
    try:
        # Extract frame at 0.5s to avoid black start frames
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return thumb_path
    except Exception as e:
        # logger.error(f"Thumb gen failed: {e}") # Logger not available here yet, need to pass it or rely on caller
        # Try at 0s if 0.5s failed (e.g. video too short)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-ss", "00:00:00", "-vframes", "1", thumb_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return thumb_path
        except:
            return None

def resolve_path(p: str):
    """
    Resolves a path that might be a URL to a local filesystem path.
    e.g. http://localhost:8000/uploads/file.png -> /abs/path/to/uploads/file.png
            /uploads/file.png -> /abs/path/to/uploads/file.png
    """
    if not p:
        return p
    p = str(p)
    
    base_dir = get_base_dir()

    # Handle full URLs first
    if "localhost" in p or "127.0.0.1" in p:
        if "/uploads/" in p:
            filename = p.split("/uploads/")[-1]
            return os.path.join(base_dir, "uploads", filename)
        if "/generated/" in p:
            filename = p.split("/generated/")[-1]
            return os.path.join(base_dir, "generated", filename)
    
    # Handle relative URL paths (starting with /uploads or /generated)
    if p.startswith("/uploads/"):
        filename = p.split("/uploads/")[-1]
        return os.path.join(base_dir, "uploads", filename)
        
    if p.startswith("/generated/"):
        filename = p.split("/generated/")[-1]
        return os.path.join(base_dir, "generated", filename)
        
    return p
