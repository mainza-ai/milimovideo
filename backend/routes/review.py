"""Frame-accurate review: comments, resolutions, and per-shot approvals."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from database import get_session, Project, ReviewItem, Shot
from schemas import ReviewApprovalRequest, ReviewCommentRequest, ReviewCommentUpdate

router = APIRouter(tags=["review"])
logger = logging.getLogger(__name__)


def _serialize(item: ReviewItem) -> dict:
    return {
        "id": item.id,
        "project_id": item.project_id,
        "shot_id": item.shot_id,
        "kind": item.kind,
        "frame": item.frame,
        "body": item.body,
        "author": item.author,
        "status": item.status,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def _project_or_404(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _check_shot(session: Session, project_id: str, shot_id: Optional[str]) -> None:
    if not shot_id:
        return
    shot = session.get(Shot, shot_id)
    if not shot or shot.project_id != project_id:
        raise HTTPException(status_code=404, detail="Shot not found in this project")


@router.get("/projects/{project_id}/review")
def list_review(project_id: str, session: Session = Depends(get_session)):
    _project_or_404(session, project_id)
    items = session.exec(
        select(ReviewItem)
        .where(ReviewItem.project_id == project_id)
        .order_by(ReviewItem.created_at.asc())
    ).all()
    comments = [_serialize(item) for item in items if item.kind == "comment"]
    approvals = [_serialize(item) for item in items if item.kind == "approval"]
    return {
        "comments": comments,
        "approvals": approvals,
        "summary": {
            "comments": len(comments),
            "open": sum(1 for item in comments if item["status"] == "open"),
            "approved": sum(1 for item in approvals if item["status"] == "approved"),
            "changes_requested": sum(
                1 for item in approvals if item["status"] == "changes_requested"
            ),
        },
    }


def _extract_memory(
    session: Session,
    project_id: str,
    text: str,
    *,
    shot_id: str | None,
    source: str,
) -> None:
    """Promote explicit review signal into project memory.

    Comments and approvals are already durable history; this makes them
    retrievable by the agent with provenance. Extraction is best-effort: a
    memory failure must never fail the review write.
    """
    if not text.strip():
        return
    try:
        from agent_memory import remember

        remember(
            session,
            project_id,
            text,
            scope="shot" if shot_id else "project",
            kind="episodic",
            shot_id=shot_id,
            provenance="user",
            trust="user",
            metadata={"source": source},
        )
    except Exception:
        logger.warning("Could not extract review memory", exc_info=True)


@router.post("/projects/{project_id}/review/comments")
def add_comment(
    project_id: str,
    req: ReviewCommentRequest,
    session: Session = Depends(get_session),
):
    _project_or_404(session, project_id)
    _check_shot(session, project_id, req.shot_id)
    body = (req.body or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail="Comment body is required")
    if req.frame < 0:
        raise HTTPException(status_code=400, detail="Frame must be zero or greater")

    item = ReviewItem(
        project_id=project_id,
        shot_id=req.shot_id,
        kind="comment",
        frame=req.frame,
        body=body[:4000],
        author=(req.author or "local")[:80],
        status="open",
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    _extract_memory(
        session,
        project_id,
        f"Review comment on shot {req.shot_id or 'project'}: {body}",
        shot_id=req.shot_id,
        source="review_comment",
    )
    return _serialize(item)


@router.patch("/projects/{project_id}/review/comments/{comment_id}")
def update_comment(
    project_id: str,
    comment_id: str,
    req: ReviewCommentUpdate,
    session: Session = Depends(get_session),
):
    _project_or_404(session, project_id)
    item = session.get(ReviewItem, comment_id)
    if not item or item.project_id != project_id or item.kind != "comment":
        raise HTTPException(status_code=404, detail="Comment not found")

    if req.body is not None:
        body = req.body.strip()
        if not body:
            raise HTTPException(status_code=400, detail="Comment body is required")
        item.body = body[:4000]
    if req.status is not None:
        item.status = req.status
    item.updated_at = datetime.utcnow()
    session.add(item)
    session.commit()
    session.refresh(item)
    if req.body is not None and item.shot_id:
        _extract_memory(
            session,
            project_id,
            f"Review comment on shot {item.shot_id}: {item.body}",
            shot_id=item.shot_id,
            source="review_comment",
        )
    return _serialize(item)


@router.delete("/projects/{project_id}/review/comments/{comment_id}")
def delete_comment(
    project_id: str,
    comment_id: str,
    session: Session = Depends(get_session),
):
    _project_or_404(session, project_id)
    item = session.get(ReviewItem, comment_id)
    if not item or item.project_id != project_id or item.kind != "comment":
        raise HTTPException(status_code=404, detail="Comment not found")
    session.delete(item)
    session.commit()
    return {"status": "deleted", "id": comment_id}


@router.post("/projects/{project_id}/review/approvals")
def set_approval(
    project_id: str,
    req: ReviewApprovalRequest,
    session: Session = Depends(get_session),
):
    _project_or_404(session, project_id)
    _check_shot(session, project_id, req.shot_id)

    item = session.exec(
        select(ReviewItem)
        .where(ReviewItem.project_id == project_id)
        .where(ReviewItem.shot_id == req.shot_id)
        .where(ReviewItem.kind == "approval")
    ).first()

    if item:
        item.status = req.status
        item.body = (req.note or "")[:4000]
        item.author = (req.author or "local")[:80]
        item.updated_at = datetime.utcnow()
    else:
        item = ReviewItem(
            project_id=project_id,
            shot_id=req.shot_id,
            kind="approval",
            frame=0,
            body=(req.note or "")[:4000],
            author=(req.author or "local")[:80],
            status=req.status,
        )
    session.add(item)
    session.commit()
    session.refresh(item)
    decision = "approved" if req.status == "approved" else "changes requested"
    note = (req.note or "").strip()
    _extract_memory(
        session,
        project_id,
        f"Shot {req.shot_id} {decision}." + (f" Note: {note}" if note else ""),
        shot_id=req.shot_id,
        source="review_approval",
    )
    return _serialize(item)
