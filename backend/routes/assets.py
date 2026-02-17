from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
import shutil
import os
import uuid
import config
from file_utils import get_base_dir
from sqlmodel import Session
from database import get_session
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["assets"])

# Global upload dir (legacy)
UPLOAD_DIR = os.path.join(get_base_dir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
def upload_file(
    file: UploadFile = File(...), 
    project_id: Optional[str] = Form(None),
    type: Optional[str] = Form("general"), # general, element, reference
    session: Session = Depends(get_session)
):
    upload_target_dir = UPLOAD_DIR
    
    # Project-scoped uploads
    if project_id:
        project_dir = os.path.join(config.PROJECTS_DIR, project_id)
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project not found")
            
        if type == "element":
            upload_target_dir = os.path.join(project_dir, "assets", "elements")
        elif type == "reference":
            upload_target_dir = os.path.join(project_dir, "assets", "references")
        else:
             upload_target_dir = os.path.join(project_dir, "assets", "general")
        
        os.makedirs(upload_target_dir, exist_ok=True)
        
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(upload_target_dir, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create DB Record
    from database import Asset
    from datetime import datetime, timezone
    
    # Calculate access URL
    if project_id:
        folder_name = "general"
        if type == "element": folder_name = "elements"
        if type == "reference": folder_name = "references"
        access_url = f"/projects/{project_id}/assets/{folder_name}/{filename}"
    else:
        access_url = f"/uploads/{filename}"

    new_asset = Asset(
        id=uuid.uuid4().hex,
        project_id=project_id,
        type="image" if file.content_type.startswith("image") else "video",
        url=access_url,
        path=file_path,
        filename=filename, # Original or safe filename? using the one on disk
        created_at=datetime.now(timezone.utc)
    )
    session.add(new_asset)
    session.commit()

    return {"url": access_url, "asset_id": new_asset.id, "type": new_asset.type, "filename": new_asset.filename, "access_path": file_path}

@router.post("/shot/{job_id}/last-frame")
async def get_shot_last_frame(job_id: str, session: Session = Depends(get_session)):
    """
    Returns the path/url to the last frame of a generated video.
    Used for 'Extend' functionality.
    """
    import subprocess
    from database import get_session, Job
    
    # Look up the job in DB to get project-scoped output path
    job = session.get(Job, job_id)
    
    if not job or not job.output_path:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or has no output")
    
    # Require project-scoped paths
    if not job.output_path.startswith("/projects/"):
        raise HTTPException(status_code=400, detail=f"Job {job_id} uses legacy path format.")
    
    # Project-scoped: /projects/{id}/generated/filename.mp4
    # We need to resolve to FS path. get_base_dir() returns backend root.
    # config.PROJECTS_DIR is absolute.
    # job.output_path looks like /projects/ID/generated/foo.mp4
    
    rel_path = job.output_path.removeprefix("/projects/")
    video_path = os.path.join(config.PROJECTS_DIR, rel_path)
    
    # Check if it's an image (single frame generation)
    if job.output_path.endswith(".jpg") or job.output_path.endswith(".png"):
        if os.path.exists(video_path):
            return {"url": job.output_path, "path": video_path}
        raise HTTPException(status_code=404, detail=f"Image not found: {job.output_path}")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    try:
        # Extract last frame to project workspace
        out_name = f"{job_id}_last.jpg"
        
        # Determine output directory (same as video)
        out_dir = os.path.dirname(video_path)
        out_path = os.path.join(out_dir, out_name)
        
        # Extract last frame using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-1.0", "-i", video_path,
            "-q:v", "2", "-update", "1", out_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Build URL based on original path pattern
        parts = job.output_path.split("/")
        project_id = parts[2]
        out_url = f"/projects/{project_id}/generated/{out_name}"
        
        return {"url": out_url, "path": out_path}
    except Exception as e:
        # logger.error(f"Failed to extract last frame from {video_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploads")
async def list_uploads(session: Session = Depends(get_session)):
    """
    Returns a combined list of Assets (Uploaded) and Completed Jobs (Generated)
    from the SQLite database, deduplicated by URL.
    """
    from database import Asset, Job
    from sqlmodel import select
    
    files = []
    seen_urls = set()  # Track URLs to avoid duplicates
    
    # 1. Get Assets (Uploaded + Generated Images) - these take priority
    assets = session.exec(select(Asset).order_by(Asset.created_at.desc())).all()
    for a in assets:
        if a.url in seen_urls:
            continue
        seen_urls.add(a.url)
        files.append({
            "id": a.id,
            "url": a.url,
            "path": a.path,
            "type": a.type,
            "filename": a.filename,
            "thumbnail": None,
            "created_at": a.created_at.timestamp()
        })
        
    # 2. Get Completed Jobs (Generated Videos) - skip if URL already seen
    jobs = session.exec(select(Job).where(Job.status == "completed").order_by(Job.created_at.desc())).all()
 

    for j in jobs:
        if not j.output_path: continue
        
        filename = os.path.basename(j.output_path)
        
        # Only serve project-scoped paths
        if j.output_path.startswith("/projects/"):
            video_url = j.output_path
        elif "/projects/" in j.output_path:
            # Handle absolute filesystem paths like PROJECTS_DIR/...
            if j.output_path.startswith(config.PROJECTS_DIR):
                rel = os.path.relpath(j.output_path, os.path.dirname(config.PROJECTS_DIR))
                video_url = f"/{rel}"
            else:
                # Skip - cannot determine correct URL
                continue
        else:
            # Skip legacy /generated/ paths - no longer supported (unless fallback needed)
            continue
        
        # Skip if already seen (e.g., image already in Assets)
        if video_url in seen_urls:
            continue
        seen_urls.add(video_url)
        
        # THUMBNAIL LOGIC
        thumb_url = None
        if j.thumbnail_path:
            thumb_url = j.thumbnail_path
        else:
            # Fallback: check filesystem (only for project paths)
            if filename.endswith(".mp4"):
                if "/projects/" in video_url:
                    try:
                        # video: /projects/ID/generated/video.mp4
                        # thumb: /projects/ID/thumbnails/video_thumb.jpg
                        parts = video_url.split("/")
                        if len(parts) >= 5 and parts[3] == "generated":
                            parts[3] = "thumbnails"
                            parts[-1] = parts[-1].replace(".mp4", "_thumb.jpg")
                            cand_url = "/".join(parts)
                            # Check existence?
                            # we assume existence if DB entry exists to save IO
                            pass 
                    except: pass
        
        files.append({
            "id": j.id,
            "url": video_url,
            "path": j.output_path,
            "type": "video" if filename.endswith(".mp4") else "image",
            "filename": filename,
            "thumbnail": thumb_url,
            "created_at": j.created_at.timestamp()
        })
        
    # Sort combined by time
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return files

@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str, session: Session = Depends(get_session)):
    from database import Asset, Job
    from sqlmodel import select
    
    # 1. Try finding Asset by ID (Primary Key)
    asset = session.get(Asset, asset_id)
    if asset:
        target_url = asset.url
        session.delete(asset)
        
        # Also clean up any Job pointing to this asset
        if target_url:
             # Try exact URL match
             jobs = session.exec(select(Job).where(Job.output_path == target_url)).all()
             
             # Fallback: Try match by filename (if URL match missed for some reason)
             if not jobs and asset.filename:
                 # Check if output_path ends with the filename
                 # SQLModel/SQLAlchemy 'endswith' or 'like'
                 jobs = session.exec(select(Job).where(Job.output_path.like(f"%/{asset.filename}"))).all()
                 
             for j in jobs:
                 j.output_path = None
                 j.status = "deleted"
                 session.add(j)
        
        session.commit()
        
        # Delete file
        if asset.path and os.path.exists(asset.path):
             try:
                 os.remove(asset.path)
             except Exception as e:
                 logger.error(f"Failed to delete asset file {asset.path}: {e}")
                 
        return {"status": "deleted", "id": asset_id}
        
    # 2. Try finding Job (if ID passed was a Job ID)
    job = session.get(Job, asset_id)
    if job:
        # Resolve absolute path from URL
        file_path = job.output_path
        target_path = job.output_path # Keep for Asset lookup
        
        if file_path and file_path.startswith("/projects/"):
            rel = file_path[len("/projects/"):]
            file_path = os.path.join(config.PROJECTS_DIR, rel)
            
        if file_path and os.path.exists(file_path):
             if os.path.isdir(file_path):
                 shutil.rmtree(file_path)
             else:
                 try: os.remove(file_path)
                 except: pass
        
        # Clean up corresponding Asset if any
        if target_path:
             assets = session.exec(select(Asset).where(Asset.url == target_path)).all()
             # Or check path? Asset.path is absolute/resolved. Asset.url is web. 
             # Job.output_path for images is usually the web URL in this codebase version?
             # Let's check Job creation in image.py: 
             # update_job_db(..., output_path=web_url...)
             # So Job stores URL. Asset stores URL and Path. 
             # So we match Job.output_path (URL) to Asset.url.
             for a in assets:
                 session.delete(a)

        job.output_path = None # Clear path
        job.status = "deleted"
        session.add(job)
        session.commit()
        return {"status": "deleted", "id": asset_id}
        
    # 3. Fallback: Try filename match (Legacy)
    # If the ID passed doesn't match a record, maybe it was a filename?
    # Only try this if it looks like a filename (has dot)
    if "." in asset_id:
        return await delete_upload(asset_id, session)

    raise HTTPException(status_code=404, detail="Asset not found")

@router.delete("/upload/{filename}")
async def delete_upload(filename: str, session: Session = Depends(get_session)):

    from database import Asset, Job
    from sqlmodel import select
    
    # Try finding Asset
    asset = session.exec(select(Asset).where(Asset.filename == filename)).first()
    if asset:
        session.delete(asset)
        session.commit()
        # Delete file
        if os.path.exists(asset.path):
             try:
                 os.remove(asset.path)
             except: pass
        return {"status": "deleted"}
        
    # Try finding Job (id usually matches filename base)
    job_id = filename.split(".")[0]
    job = session.get(Job, job_id)
    if job:
        # Resolve absolute path from URL
        file_path = job.output_path
        if file_path and file_path.startswith("/projects/"):
            rel = file_path.removeprefix("/projects/")
            # Safer slice
            rel = file_path[len("/projects/"):]
            file_path = os.path.join(config.PROJECTS_DIR, rel)
            
        if file_path and os.path.exists(file_path):
             if os.path.isdir(file_path):
                 shutil.rmtree(file_path)
             else:
                 try: os.remove(file_path)
                 except: pass
        
        job.output_path = None # Clear path
        job.status = "deleted"
        session.add(job)
        session.commit()
        return {"status": "deleted"}
        
    raise HTTPException(status_code=404, detail="File not found")
