"""Project-scoped agent memory API: remember, search, pack, approve, forget."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session

from agent_memory import (
    DEFAULT_BUDGET_TOKENS,
    KINDS,
    MAX_TEXT_CHARS,
    PROVENANCE,
    SCOPES,
    TRUST_LEVELS,
    approve,
    clear_project,
    context_pack,
    export_project,
    forget,
    remember,
    search,
    update,
)
from database import get_session

router = APIRouter()


class MemoryCreate(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    scope: Literal["project", "shot", "session", "agent"] = "project"
    kind: Literal["working", "episodic", "semantic", "procedural"] = "semantic"
    shot_id: Optional[str] = None
    provenance: Literal["user", "agent", "imported", "system"] = "user"
    trust: Literal["user", "verified", "agent", "imported"] = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)
    pinned: bool = False


class MemoryUpdate(BaseModel):
    text: Optional[str] = Field(default=None, max_length=MAX_TEXT_CHARS)
    pinned: Optional[bool] = None
    trust: Optional[Literal["user", "verified", "agent", "imported"]] = None
    kind: Optional[Literal["working", "episodic", "semantic", "procedural"]] = None
    metadata: Optional[dict[str, Any]] = None


@router.get("/projects/{project_id}/memory")
def list_memory(
    project_id: str,
    query: str = "",
    kind: Optional[str] = None,
    scope: Optional[str] = None,
    shot_id: Optional[str] = None,
    include_quarantined: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    if kind and kind not in KINDS:
        raise HTTPException(status_code=422, detail=f"unknown kind {kind!r}")
    if scope and scope not in SCOPES:
        raise HTTPException(status_code=422, detail=f"unknown scope {scope!r}")
    return {
        "project_id": project_id,
        "entries": search(
            session,
            project_id,
            query,
            kind=kind,
            scope=scope,
            shot_id=shot_id,
            include_quarantined=include_quarantined,
            limit=limit,
        ),
    }


@router.post("/projects/{project_id}/memory")
def create_memory(
    project_id: str,
    payload: MemoryCreate,
    session: Session = Depends(get_session),
):
    if payload.provenance not in PROVENANCE:
        raise HTTPException(status_code=422, detail="unknown provenance")
    if payload.trust not in TRUST_LEVELS:
        raise HTTPException(status_code=422, detail="unknown trust level")
    try:
        entry = remember(
            session,
            project_id,
            payload.text,
            scope=payload.scope,
            kind=payload.kind,
            shot_id=payload.shot_id,
            provenance=payload.provenance,
            trust=payload.trust,
            metadata=payload.metadata,
            pinned=payload.pinned,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return entry


@router.get("/projects/{project_id}/memory/context")
def memory_context(
    project_id: str,
    query: str = "",
    budget_tokens: int = Query(default=DEFAULT_BUDGET_TOKENS, ge=64, le=8000),
    shot_id: Optional[str] = None,
    session: Session = Depends(get_session),
):
    return context_pack(
        session,
        project_id,
        query,
        budget_tokens=budget_tokens,
        shot_id=shot_id,
    )


@router.get("/projects/{project_id}/memory/export")
def export_memory(project_id: str, session: Session = Depends(get_session)):
    return export_project(session, project_id)


@router.delete("/projects/{project_id}/memory")
def clear_memory(project_id: str, session: Session = Depends(get_session)):
    deleted = clear_project(session, project_id)
    return {"project_id": project_id, "deleted": deleted}


@router.patch("/projects/{project_id}/memory/{memory_id}")
def update_memory(
    project_id: str,
    memory_id: str,
    payload: MemoryUpdate,
    session: Session = Depends(get_session),
):
    try:
        entry = update(
            session,
            project_id,
            memory_id,
            text_value=payload.text,
            pinned=payload.pinned,
            trust=payload.trust,
            kind=payload.kind,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if entry is None:
        raise HTTPException(status_code=404, detail="memory entry not found")
    return entry


@router.post("/projects/{project_id}/memory/{memory_id}/approve")
def approve_memory(
    project_id: str,
    memory_id: str,
    session: Session = Depends(get_session),
):
    entry = approve(session, project_id, memory_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="memory entry not found")
    return entry


@router.delete("/projects/{project_id}/memory/{memory_id}")
def forget_memory(
    project_id: str,
    memory_id: str,
    session: Session = Depends(get_session),
):
    if not forget(session, project_id, memory_id):
        raise HTTPException(status_code=404, detail="memory entry not found")
    return {"project_id": project_id, "deleted": memory_id}
