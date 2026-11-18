"""ComfyUI bridge endpoints: readiness, validation, and queued workflow runs."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from database import Job, Project, Shot, get_session
from schemas import ComfyRunRequest, ComfyWorkflowRequest
import config
from services.comfy import (
    ComfyClient,
    ComfyError,
    apply_overrides,
    parse_workflow,
    validate_workflow,
    workflow_size,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["comfy"])


def _require_enabled() -> None:
    if not config.COMFY_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="ComfyUI bridge is disabled (MILIMO_COMFY_ENABLED=0)",
        )


@router.get("/comfy/health")
def comfy_health():
    """Probe the optional ComfyUI server without failing the UI when it is absent."""
    if not config.COMFY_ENABLED:
        return {
            "enabled": False,
            "reachable": False,
            "url": config.COMFY_URL,
            "error": "ComfyUI bridge disabled (MILIMO_COMFY_ENABLED=0)",
        }
    with ComfyClient() as client:
        return {"enabled": True, **client.health()}


@router.post("/comfy/validate")
def comfy_validate(req: ComfyWorkflowRequest):
    """Structural validation for the workflow editor; no ComfyUI call."""
    try:
        workflow = apply_overrides(parse_workflow(req.workflow), req.overrides)
    except ComfyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "enabled": config.COMFY_ENABLED,
        **validate_workflow(workflow),
    }


@router.post("/comfy/free")
def comfy_free(session: Session = Depends(get_session)):
    """Ask ComfyUI to unload cached models and release our residency slot."""
    _require_enabled()
    from memory_manager import memory_manager

    slot = (memory_manager.status().get("slots") or {}).get("comfy") or {}
    if slot.get("leases", 0) > 0:
        raise HTTPException(status_code=409, detail="A ComfyUI workflow is running")
    with ComfyClient() as client:
        health = client.health()
        if not health.get("reachable"):
            raise HTTPException(status_code=503, detail="ComfyUI is not reachable")
        if health.get("queue_running"):
            raise HTTPException(status_code=409, detail="ComfyUI is executing a workflow")
        try:
            result = client.free(unload_models=True)
        except ComfyError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    memory_manager.release("comfy")
    return result


@router.post("/comfy/run")
async def comfy_run(
    req: ComfyRunRequest,
    session: Session = Depends(get_session),
):
    """Queue a ComfyUI workflow and ingest every output file it writes."""
    _require_enabled()
    project = session.get(Project, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if req.shot_id:
        shot = session.get(Shot, req.shot_id)
        if not shot or shot.project_id != req.project_id:
            raise HTTPException(status_code=404, detail="Shot not found in this project")

    try:
        workflow = apply_overrides(parse_workflow(req.workflow), req.overrides)
    except ComfyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    validation = validate_workflow(workflow)
    if not validation["output_nodes"]:
        raise HTTPException(
            status_code=400,
            detail="Workflow has no Save/Preview node, so it would produce no files.",
        )

    width, height = workflow_size(workflow)
    job_id = f"job_comfy_{uuid.uuid4().hex[:8]}"
    params = {
        "project_id": req.project_id,
        "shot_id": req.shot_id,
        "workflow": workflow,
        "overrides": {},
        "filename_prefix": req.filename_prefix,
        "prompt": req.prompt,
        "estimate_gb": req.estimate_gb,
        "estimate_seconds": req.estimate_seconds,
        "timeout_sec": req.timeout_sec,
        "width": width,
        "height": height,
    }
    job = Job(
        id=job_id,
        project_id=req.project_id,
        shot_id=req.shot_id,
        type="comfy",
        status="pending",
        created_at=datetime.now(timezone.utc),
        prompt=req.prompt or "ComfyUI workflow",
        params_json=json.dumps(params),
    )
    session.add(job)
    session.commit()

    from job_utils import enqueue_gpu_job

    await enqueue_gpu_job(job_id, "comfy", params)
    return {
        "job_id": job_id,
        "status": "queued",
        "prompt_id": None,
        "validation": validation,
        "comfy_url": config.COMFY_URL,
    }
