from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session
from database import get_session, Shot, Project, Job
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime
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
        })
    return {"shot_id": shot_id, "takes": takes}


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
