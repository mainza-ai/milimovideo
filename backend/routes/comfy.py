"""ComfyUI bridge endpoints: readiness, validation, and queued workflow runs."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from database import ComfyWorkflow, Job, Project, Shot, get_session
from schemas import (
    ComfyRunRequest,
    ComfySavedWorkflowRequest,
    ComfySettingsRequest,
    ComfyUploadRequest,
    ComfyWorkflowRequest,
)
import config
from services.comfy_settings import ComfyRevisionConflict, ComfySettingsError, comfy_settings
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


def _effective_settings() -> dict:
    return comfy_settings.effective_values()


def _require_enabled() -> dict:
    settings = _effective_settings()
    if not settings["enabled"]:
        raise HTTPException(
            status_code=409,
            detail="ComfyUI bridge is disabled (MILIMO_COMFY_ENABLED=0 or Setup)",
        )
    return settings


@router.get("/comfy/health")
def comfy_health():
    """Probe the optional ComfyUI server without failing the UI when it is absent."""
    settings = _effective_settings()
    if not settings["enabled"]:
        return {
            "enabled": False,
            "reachable": False,
            "url": settings["url"],
            "hard_disabled": not config.COMFY_ENABLED,
            "declared_gb": settings["declared_gb"],
            "has_token": bool(settings.get("token")),
            "error": "ComfyUI bridge disabled (MILIMO_COMFY_ENABLED=0 or Setup)",
        }
    with ComfyClient() as client:
        return {
            "enabled": True,
            **client.health(),
            "declared_gb": settings["declared_gb"],
            "has_token": bool(settings.get("token")),
        }


@router.post("/comfy/validate")
def comfy_validate(req: ComfyWorkflowRequest):
    """Structural validation for the workflow editor; no ComfyUI call."""
    try:
        workflow = _clean_workflow(apply_overrides(parse_workflow(req.workflow), req.overrides))
    except ComfyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    structural = validate_workflow(workflow)
    if not req.live:
        return {"ok": True, "enabled": _effective_settings()["enabled"], "live": False, **structural}
    settings = _effective_settings()
    if not settings["enabled"]:
        return {
            "ok": False, "enabled": False, "live": False,
            "error": "ComfyUI bridge is disabled",
            **structural,
        }
    try:
        with ComfyClient() as client:
            live = client.validate_live(workflow)
    except ComfyError as exc:
        return {
            "ok": False, "enabled": True, "live": False,
            "error": str(exc),
            **structural,
        }
    return {"enabled": True, "live": True, **live}


@router.get("/comfy/settings")
def comfy_get_settings():
    return comfy_settings.public()


@router.put("/comfy/settings")
def comfy_update_settings(req: ComfySettingsRequest):
    payload = req.model_dump(exclude_none=True)
    expected = payload.pop("expected_revision", None)
    try:
        return comfy_settings.save(payload, expected_revision=expected)
    except ComfyRevisionConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "revision": exc.current, "settings": comfy_settings.public()},
        ) from exc
    except ComfySettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/comfy/catalog")
def comfy_catalog():
    _require_enabled()
    try:
        with ComfyClient() as client:
            return client.catalog()
    except ComfyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/comfy/schema")
def comfy_schema(classes: str = Query(..., min_length=1)):
    _require_enabled()
    names = [part.strip() for part in classes.split(",") if part.strip()][:100]
    try:
        with ComfyClient() as client:
            return {"classes": client.schema_for(names)}
    except ComfyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/comfy/models")
def comfy_models(folder: str = Query(..., min_length=1)):
    _require_enabled()
    allowed = {
        "checkpoints", "diffusion_models", "unet", "loras", "vae",
        "text_encoders", "clip", "clip_vision", "controlnet",
        "upscale_models", "latent_upscale_models", "embeddings",
    }
    if folder not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported ComfyUI model folder")
    try:
        with ComfyClient() as client:
            return {"folder": folder, "models": client.models(folder)}
    except ComfyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/comfy/upload")
def comfy_upload(req: ComfyUploadRequest):
    _require_enabled()
    from file_utils import resolve_path

    path = resolve_path(req.url)
    root = os.path.realpath(config.PROJECTS_DIR)
    path = os.path.realpath(path) if path else None
    if not path or not path.startswith(root + os.sep):
        raise HTTPException(status_code=400, detail="Only project media can be uploaded")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Media file not found")
    if os.path.getsize(path) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Media exceeds the upload limit")
    filename = os.path.basename(path)
    try:
        with open(path, "rb") as handle:
            content = handle.read()
        with ComfyClient() as client:
            result = client.upload_image(filename, content)
    except ComfyError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {**result, "source_url": req.url}


def _workflow_row(row: ComfyWorkflow) -> dict:
    try:
        workflow = json.loads(row.graph_json or "{}")
    except json.JSONDecodeError:
        workflow = {}
    try:
        bindings = json.loads(row.bindings_json or "{}")
    except json.JSONDecodeError:
        bindings = {}
    return {
        "id": row.id,
        "project_id": row.project_id,
        "name": row.name,
        "description": row.description,
        "workflow": workflow,
        "bindings": bindings,
        "revision": row.revision,
        "is_default": bool(row.is_default),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _clean_workflow(workflow) -> dict:
    parsed = parse_workflow(workflow)
    if len(parsed) > config.COMFY_MAX_NODES:
        raise HTTPException(status_code=400, detail=f"Workflow has too many nodes (max {config.COMFY_MAX_NODES})")
    encoded = json.dumps(parsed)
    if len(encoded.encode()) > config.COMFY_MAX_WORKFLOW_BYTES:
        raise HTTPException(status_code=400, detail="Workflow JSON is too large")
    return parsed


@router.get("/projects/{project_id}/comfy/workflows")
def list_comfy_workflows(project_id: str, session: Session = Depends(get_session)):
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    rows = session.exec(
        select(ComfyWorkflow)
        .where(ComfyWorkflow.project_id == project_id)
        .order_by(ComfyWorkflow.is_default.desc(), ComfyWorkflow.name)
    ).all()
    return [_workflow_row(row) for row in rows]


@router.post("/projects/{project_id}/comfy/workflows")
def create_comfy_workflow(
    project_id: str,
    req: ComfySavedWorkflowRequest,
    session: Session = Depends(get_session),
):
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workflow name is required")
    workflow = _clean_workflow(req.workflow)
    row = ComfyWorkflow(
        project_id=project_id,
        name=name[:100],
        description=req.description[:400],
        graph_json=json.dumps(workflow),
        bindings_json=json.dumps(req.bindings),
        revision=0,
        is_default=bool(req.is_default),
    )
    if row.is_default:
        for other in session.exec(select(ComfyWorkflow).where(ComfyWorkflow.project_id == project_id)).all():
            other.is_default = False
            session.add(other)
    session.add(row)
    session.commit()
    session.refresh(row)
    return _workflow_row(row)


@router.put("/projects/{project_id}/comfy/workflows/{workflow_id}")
def update_comfy_workflow(
    project_id: str,
    workflow_id: str,
    req: ComfySavedWorkflowRequest,
    session: Session = Depends(get_session),
):
    row = session.get(ComfyWorkflow, workflow_id)
    if not row or row.project_id != project_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    expected = req.expected_revision if req.expected_revision is not None else row.revision
    if expected != row.revision:
        raise HTTPException(status_code=409, detail={"message": "Workflow revision conflict", "revision": row.revision, "workflow": _workflow_row(row)})
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workflow name is required")
    workflow = _clean_workflow(req.workflow)
    if req.is_default:
        for other in session.exec(select(ComfyWorkflow).where(ComfyWorkflow.project_id == project_id)).all():
            if other.id != row.id:
                other.is_default = False
                session.add(other)
    row.name = name[:100]
    row.description = req.description[:400]
    row.graph_json = json.dumps(workflow)
    row.bindings_json = json.dumps(req.bindings)
    row.is_default = bool(req.is_default)
    row.revision += 1
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()
    session.refresh(row)
    return _workflow_row(row)


@router.delete("/projects/{project_id}/comfy/workflows/{workflow_id}")
def delete_comfy_workflow(project_id: str, workflow_id: str, session: Session = Depends(get_session)):
    row = session.get(ComfyWorkflow, workflow_id)
    if not row or row.project_id != project_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    session.delete(row)
    session.commit()
    return {"status": "deleted", "id": workflow_id}


@router.post("/projects/{project_id}/comfy/workflows/{workflow_id}/default")
def default_comfy_workflow(project_id: str, workflow_id: str, session: Session = Depends(get_session)):
    row = session.get(ComfyWorkflow, workflow_id)
    if not row or row.project_id != project_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    for other in session.exec(select(ComfyWorkflow).where(ComfyWorkflow.project_id == project_id)).all():
        other.is_default = other.id == row.id
        session.add(other)
    session.commit()
    session.refresh(row)
    return _workflow_row(row)


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
        workflow = _clean_workflow(apply_overrides(parse_workflow(req.workflow), req.overrides))
    except ComfyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    validation = validate_workflow(workflow)
    if not validation["output_nodes"]:
        raise HTTPException(
            status_code=400,
            detail="Workflow has no Save/Preview node, so it would produce no files.",
        )

    width, height = workflow_size(workflow)
    from database import QueuedJob

    active = session.exec(
        select(QueuedJob)
        .where(QueuedJob.kind == "comfy")
        .where(QueuedJob.status.in_(["queued", "running"]))
    ).all()
    for row in active:
        try:
            active_params = json.loads(row.params_json or "{}")
        except json.JSONDecodeError:
            continue
        if active_params.get("shot_id") == req.shot_id and req.shot_id:
            raise HTTPException(
                status_code=409,
                detail={"message": "A ComfyUI workflow is already active for this shot", "job_id": row.job_id},
            )

    settings = _effective_settings()
    job_id = f"job_comfy_{uuid.uuid4().hex[:8]}"
    params = {
        "project_id": req.project_id,
        "shot_id": req.shot_id,
        "workflow": workflow,
        "overrides": {},
        "filename_prefix": req.filename_prefix,
        "prompt": req.prompt,
        "workflow_id": req.workflow_id,
        "workflow_name": req.workflow_name,
        "bindings": req.bindings,
        "estimate_gb": req.estimate_gb if req.estimate_gb is not None else settings["declared_gb"],
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
