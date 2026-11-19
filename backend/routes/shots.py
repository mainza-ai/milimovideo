from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlmodel import Session
from database import get_session, Shot, Project, Job
from typing import Optional, List, Dict, Any
import logging
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
import config
from file_utils import resolve_path
from sqlmodel import select
from schemas import TimelineItem, SelectTakeRequest, FavoriteTakeRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["shots"])

@router.patch("/shots/{shot_id}")
async def update_shot_partial(shot_id: str, updates: dict, session: Session = Depends(get_session)):
    """
    Granular update for a single shot. 
    Handles updates to timeline, params, and metadata.
    Does NOT handle splitting or major structural changes (use dedicated endpoints).
    """
    shot = session.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
        
    # Allowed fields to patch
    # We filter out ID and Foreign Keys to prevent accidental corruption
    protected_fields = {"id", "project_id", "scene_id", "created_at"}
    
    for k, v in updates.items():
        if k in protected_fields:
            continue
            
        if hasattr(shot, k):
            if k == "timeline":
                setattr(shot, k, json.dumps(TimelineItem.normalize(v)))
            elif k == "keyframes":
                setattr(shot, k, json.dumps(v) if isinstance(v, (list, dict)) else v)
            elif k == "blocking":
                setattr(shot, k, json.dumps(v) if isinstance(v, (list, dict)) else v)
            elif k == "matched_elements":
                setattr(shot, k, json.dumps(v) if isinstance(v, (list, dict)) else v)
            else:
                setattr(shot, k, v)

    project = session.get(Project, shot.project_id)
    if project:
        project.updated_at = datetime.utcnow()
        project.revision = (project.revision or 0) + 1
        session.add(project)

    session.add(shot)
    session.commit()
    session.refresh(shot)
    return shot


@router.get("/shots/{shot_id}/takes")
def list_takes(shot_id: str, session: Session = Depends(get_session)):
    """Immutable take history from completed generation jobs.

    No separate table is needed: the job ledger already owns model, seed, params,
    output path, thumbnail, and timestamp. Selecting a take only repoints the shot.
    """
    shot = session.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    jobs = session.exec(
        select(Job)
        .where(Job.shot_id == shot_id)
        .where(Job.status == "completed")
        .where(Job.output_path.isnot(None))
        .where(Job.type.in_(["shot_generation", "advanced", "video", "image", "inpaint"]))
        .order_by(Job.created_at.desc())
    ).all()

    selected_id = shot.selected_take_id or shot.last_job_id
    takes = []
    for job in jobs:
        resolved = resolve_path(job.output_path)
        missing = not bool(resolved and os.path.exists(resolved))
        params = {}
        try:
            params = json.loads(job.params_json or "{}")
        except json.JSONDecodeError:
            pass
        takes.append({
            "job_id": job.id,
            "url": job.output_path,
            "thumbnail_url": job.thumbnail_path,
            "type": job.type,
            "prompt": job.prompt,
            "enhanced_prompt": job.enhanced_prompt,
            "seed": params.get("seed"),
            "model": params.get("pipeline_override") or params.get("pipeline_type"),
            "num_frames": params.get("num_frames"),
            "llm": params.get("llm"),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "selected": job.id == selected_id,
            "favorite": job.id == shot.favorite_take_id,
            "missing": missing,
        })
    return {"shot_id": shot_id, "takes": takes}


@router.post("/shots/{shot_id}/takes/{job_id}/replace")
def replace_take(
    shot_id: str,
    job_id: str,
    file: UploadFile = File(...),
    purge_old: bool = Form(False),
    session: Session = Depends(get_session),
):
    """Relink a missing take in place, keeping its job/take identity."""
    from database import Asset
    from routes.assets import _make_video_thumbnail, _thumbnail_target

    shot = session.get(Shot, shot_id)
    job = session.get(Job, job_id)
    if not shot or not job or job.shot_id != shot_id:
        raise HTTPException(status_code=404, detail="Take not found")
    if job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=400, detail="Take has no completed media")

    original_name = Path(file.filename or "upload").name
    original_name = re.sub(r"[\x00-\x1f]", "", original_name).strip()
    extension = Path(original_name).suffix.lower()
    if not original_name or extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    old_url = job.output_path
    old_path = resolve_path(old_url)
    old_video = (job.thumbnail_path is not None) or Path(old_url).suffix.lower() in {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
    new_video = extension not in image_extensions
    if old_video != new_video:
        raise HTTPException(
            status_code=409,
            detail="Replace a take with the same media kind (video for video, image for image)",
        )

    folder = os.path.dirname(old_path) if old_path else os.path.join(config.PROJECTS_DIR, shot.project_id, "generated")
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}_{Path(original_name).stem[:120]}{extension}"
    file_path = os.path.join(folder, filename)
    temp_path = f"{file_path}.part"
    written = 0
    try:
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {config.MAX_UPLOAD_MB} MB limit")
                buffer.write(chunk)
        os.replace(temp_path, file_path)
    except Exception:
        for path in (temp_path, file_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise
    finally:
        file.file.close()

    rel = os.path.relpath(file_path, config.PROJECTS_DIR)
    if rel.startswith(".."):
        new_url = f"/uploads/{filename}"
    else:
        new_url = "/projects/" + rel.replace(os.sep, "/")

    thumbnail_url = None
    if new_video:
        thumb_path, thumb_url = _thumbnail_target(shot.project_id, filename)
        if _make_video_thumbnail(file_path, thumb_path):
            thumbnail_url = thumb_url

    job.output_path = new_url
    job.thumbnail_path = thumbnail_url
    session.add(job)

    if shot.selected_take_id == job.id or shot.last_job_id == job.id:
        shot.video_url = new_url
        shot.thumbnail_url = thumbnail_url
        shot.status = "completed"
        session.add(shot)

    # Keep an Asset row that pointed at this take in sync.
    assets = session.exec(select(Asset).where(Asset.url == old_url)).all()
    for asset in assets:
        asset.url = new_url
        asset.path = file_path
        asset.filename = filename
        asset.thumbnail_url = thumbnail_url
        asset.type = "video" if new_video else "image"
        session.add(asset)

    project = session.get(Project, shot.project_id)
    if project:
        project.updated_at = datetime.utcnow()
        project.revision = (project.revision or 0) + 1
        session.add(project)

    removed_old = False
    if purge_old and old_path and old_path != file_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
            removed_old = True
        except OSError:
            removed_old = False

    session.commit()
    session.refresh(job)
    return {
        "job_id": job.id,
        "url": job.output_path,
        "thumbnail_url": job.thumbnail_path,
        "type": job.type,
        "selected": shot.selected_take_id == job.id,
        "favorite": shot.favorite_take_id == job.id,
        "missing": False,
        "old_url": old_url,
        "old_path": old_path,
        "old_removed": removed_old,
    }


@router.post("/shots/{shot_id}/select-take")
def select_take(
    shot_id: str,
    req: SelectTakeRequest,
    session: Session = Depends(get_session),
):
    shot = session.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    job = session.get(Job, req.job_id)
    if not job or job.shot_id != shot_id or job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=400, detail="Take is not a completed output for this shot")

    shot.selected_take_id = job.id
    shot.last_job_id = job.id
    shot.video_url = job.output_path
    if job.thumbnail_path:
        shot.thumbnail_url = job.thumbnail_path
    shot.status = "completed"

    project = session.get(Project, shot.project_id)
    if project:
        project.updated_at = datetime.utcnow()
        project.revision = (project.revision or 0) + 1
        session.add(project)
    session.add(shot)
    session.commit()
    session.refresh(shot)
    return {"status": "selected", "shot_id": shot.id, "job_id": job.id, "url": shot.video_url}


@router.post("/shots/{shot_id}/favorite-take")
def favorite_take(
    shot_id: str,
    req: FavoriteTakeRequest,
    session: Session = Depends(get_session),
):
    shot = session.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    if req.job_id:
        job = session.get(Job, req.job_id)
        if not job or job.shot_id != shot_id or job.status != "completed" or not job.output_path:
            raise HTTPException(status_code=400, detail="Take is not a completed output for this shot")
        shot.favorite_take_id = job.id
    else:
        shot.favorite_take_id = None

    project = session.get(Project, shot.project_id)
    if project:
        project.updated_at = datetime.utcnow()
        project.revision = (project.revision or 0) + 1
        session.add(project)
    session.add(shot)
    session.commit()
    return {"status": "saved", "shot_id": shot.id, "favorite_take_id": shot.favorite_take_id}
