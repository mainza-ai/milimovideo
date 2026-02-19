from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session
from database import get_session, Shot, Project
from typing import Optional, List, Dict, Any
import logging
import json

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
            # Special handling for JSON fields if needed
            if k in ["timeline", "matched_elements"] and isinstance(v, (list, dict)):
                # Ensure it's stored as JSON string
                setattr(shot, k, json.dumps(v))
            else:
                setattr(shot, k, v)
            
    session.add(shot)
    session.commit()
    session.refresh(shot)
    return shot
