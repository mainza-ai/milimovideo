"""Measured memory budget, preflight, and manual residency controls."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session, select
from database import Job, get_session
import json
from pydantic import BaseModel, Field

from memory_manager import memory_manager
from services.memory import budget, children, engines, sampler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


class PreflightRequest(BaseModel):
    kind: str = "video"
    params: dict = Field(default_factory=dict)


class UnloadRequest(BaseModel):
    slot: str | None = None
    provider_id: str | None = None
    all: bool = False


class KeepWarmRequest(BaseModel):
    slot: str
    keep_warm: bool = True


@router.get("/memory")
def memory_status(session: Session = Depends(get_session)):
    snapshot = memory_manager.status()
    snapshot["sampler"] = sampler.current()
    snapshot["events"] = __import__("events", fromlist=["event_manager"]).event_manager.stats()
    recent = session.exec(
        select(Job)
        .where(Job.memory_json.isnot(None))
        .order_by(Job.created_at.desc())
        .limit(8)
    ).all()
    rows = []
    for job in recent:
        try:
            trace = json.loads(job.memory_json or "{}")
        except json.JSONDecodeError:
            continue
        rows.append({
            "job_id": job.id,
            "type": job.type,
            "status": job.status,
            "created_at": job.created_at.timestamp() if job.created_at else None,
            "peak_gb": trace.get("peak_gb"),
            "peak_stage": trace.get("peak_stage"),
            "stages": trace.get("stages") or {},
            "physical_min_gb": trace.get("physical_min_gb"),
            "swap_peak_percent": trace.get("swap_peak_percent"),
        })
    snapshot["recent_jobs"] = rows
    child_rows = children.snapshot(force=True)
    snapshot["children"] = child_rows["summary"]
    snapshot["child_processes"] = child_rows["processes"]
    return snapshot


@router.post("/memory/preflight")
def memory_preflight(req: PreflightRequest):
    decision = budget.admit(req.kind, req.params)
    return decision.as_dict()


@router.post("/memory/unload")
def memory_unload(req: UnloadRequest):
    if req.all:
        slots = memory_manager.unload_all()
        result = {"slots": slots, "engines": engines.unload()}
    elif req.provider_id:
        result = {"engines": engines.unload(provider_id=req.provider_id)}
    elif req.slot:
        if req.slot not in memory_manager.status().get("slots", {}):
            raise HTTPException(404, detail=f"slot {req.slot!r} is not resident")
        memory_manager.unload_slot(req.slot)
        result = {"slots": [req.slot]}
    else:
        raise HTTPException(400, detail="provide slot, provider_id, or all=true")
    logger.info("Memory unload: %s", result)
    return {"result": result, "memory": memory_manager.status()}


@router.post("/memory/keep-warm")
def memory_keep_warm(req: KeepWarmRequest):
    if not memory_manager.set_keep_warm(req.slot, req.keep_warm):
        raise HTTPException(404, detail=f"slot {req.slot!r} is not resident")
    return memory_manager.status()
