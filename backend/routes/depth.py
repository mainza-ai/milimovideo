"""Queued local depth estimation for stage overlays."""

import json
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from database import Job, get_session
from schemas import DepthRequest

router = APIRouter(tags=["depth"])


@router.post("/projects/{project_id}/depth")
async def estimate_depth(
    project_id: str,
    req: DepthRequest,
    session: Session = Depends(get_session),
):
    from services.model_registry import model_registry

    model = model_registry.get_model_info("depth-anything-v2-small")
    if not model or not os.path.exists(model.absolute_path):
        raise HTTPException(
            status_code=409,
            detail="Depth Anything V2 Small is not downloaded. Download it in Setup.",
        )

    job_id = f"job_depth_{uuid.uuid4().hex[:8]}"
    params = {
        "project_id": project_id,
        "shot_id": req.shot_id,
        "image_path": req.image_path,
    }
    job = Job(
        id=job_id,
        project_id=project_id,
        shot_id=req.shot_id,
        type="depth",
        status="pending",
        created_at=datetime.now(timezone.utc),
        prompt="Depth estimate",
        params_json=json.dumps(params),
    )
    session.add(job)
    session.commit()

    from job_utils import enqueue_gpu_job
    await enqueue_gpu_job(job_id, "depth", params)
    return {"job_id": job_id, "status": "queued"}
