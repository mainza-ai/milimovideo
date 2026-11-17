"""Measured memory budget, preflight, and manual residency controls."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from memory_manager import memory_manager
from services.memory import budget, engines, sampler

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
def memory_status():
    snapshot = memory_manager.status()
    snapshot["sampler"] = sampler.current()
    snapshot["events"] = __import__("events", fromlist=["event_manager"]).event_manager.stats()
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
