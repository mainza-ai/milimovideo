from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session
from database import get_session, Shot, Project
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime
from schemas import TimelineItem

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
