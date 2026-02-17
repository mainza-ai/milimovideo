import os
import shutil
import logging
import subprocess
from typing import Optional
from urllib.parse import urlparse, unquote
import config

logger = logging.getLogger(__name__)

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
        logger.warning(f"Thumb gen failed at 0.5s: {e}")
        # Try at 0s if 0.5s failed (e.g. video too short)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-ss", "00:00:00", "-vframes", "1", thumb_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return thumb_path
        except:
            return None

def resolve_path(p: str) -> str:
    """
    Resolves a path that might be a URL to a local filesystem path.
    Handles:
      - Full URLs: http://localhost:8000/projects/id/generated/file.mp4
      - Project paths: /projects/{id}/generated/file.mp4
      - Legacy paths: /uploads/file.png, /generated/file.mp4
      - Absolute paths: returned as-is
      - URL-encoded characters: %20 → space, etc.
    """
    if not p:
        return p
    p = str(p)
    
    base_dir = get_base_dir()

    # Strip full URL to just the path component
    if p.startswith("http://") or p.startswith("https://"):
        parsed = urlparse(p)
        p = unquote(parsed.path)  # URL-decode the path
    else:
        p = unquote(p)  # URL-decode even relative paths
    
    # Already an absolute filesystem path
    if os.path.isabs(p) and not p.startswith("/projects/") and not p.startswith("/uploads/") and not p.startswith("/generated/"):
        return p

    # --- Project paths: /projects/{project_id}/... ---
    if p.startswith("/projects/"):
        relative = p[len("/projects/"):]  # Safe prefix removal (not lstrip!)
        full_path = os.path.join(config.PROJECTS_DIR, relative)
        if os.path.exists(full_path):
            return full_path
        # Return even if not found — let caller handle missing file
        logger.debug(f"resolve_path: /projects/ path resolved to {full_path} (exists={os.path.exists(full_path)})")
        return full_path

    # --- Legacy paths: /uploads/... ---
    if p.startswith("/uploads/"):
        filename = p[len("/uploads/"):]
        return os.path.join(base_dir, "uploads", filename)

    # --- Legacy paths: /generated/... ---
    if p.startswith("/generated/"):
        filename = p[len("/generated/"):]
        return os.path.join(base_dir, "generated", filename)

    # Fallback — return as-is
    return p
