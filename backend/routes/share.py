"""Local capability links for reviewers.

A share link can read the project and write comments/approvals. It cannot edit
shots, trigger renders, or touch models. Only a hash of the token is stored.
Share links are disabled when the API token gate is on, because a reviewer would
not have the studio token needed to fetch media.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

import config
from database import get_session, Project, ReviewItem, ShareLink, Shot
from schemas import ShareApprovalRequest, ShareCommentRequest

router = APIRouter(tags=["share"])


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _serialize(item: ReviewItem) -> dict:
    return {
        "id": item.id,
        "shot_id": item.shot_id,
        "kind": item.kind,
        "frame": item.frame,
        "body": item.body,
        "author": item.author,
        "status": item.status,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def _share_or_404(session: Session, token: str) -> ShareLink:
    link = session.exec(
        select(ShareLink).where(ShareLink.token_hash == _hash_token(token))
    ).first()
    if not link or link.revoked:
        raise HTTPException(status_code=404, detail="Share link not found or revoked")
    return link


def _check_shot(session: Session, project_id: str, shot_id: Optional[str]) -> None:
    if not shot_id:
        return
    shot = session.get(Shot, shot_id)
    if not shot or shot.project_id != project_id:
        raise HTTPException(status_code=404, detail="Shot not found in this project")


@router.post("/projects/{project_id}/share")
def create_share(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if config.API_TOKEN:
        raise HTTPException(
            status_code=409,
            detail=(
                "Share links are disabled while MILIMO_API_TOKEN is set: a reviewer "
                "would not have the studio token required to load media."
            ),
        )

    token = secrets.token_urlsafe(24)
    link = ShareLink(project_id=project_id, token_hash=_hash_token(token))
    session.add(link)
    session.commit()
    session.refresh(link)
    return {
        "share_id": link.id,
        "token": token,
        "path": f"/studio/share/{token}",
    }


@router.delete("/projects/{project_id}/share/{share_id}")
def revoke_share(
    project_id: str,
    share_id: str,
    session: Session = Depends(get_session),
):
    link = session.get(ShareLink, share_id)
    if not link or link.project_id != project_id:
        raise HTTPException(status_code=404, detail="Share link not found")
    link.revoked = True
    session.add(link)
    session.commit()
    return {"status": "revoked", "share_id": share_id}


@router.get("/share/{token}")
def read_share(token: str, session: Session = Depends(get_session)):
    link = _share_or_404(session, token)
    project = session.get(Project, link.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    shots = session.exec(
        select(Shot)
        .where(Shot.project_id == project.id)
        .order_by(Shot.sequence_index.is_(None), Shot.sequence_index, Shot.created_at, Shot.id)
    ).all()
    items = session.exec(
        select(ReviewItem)
        .where(ReviewItem.project_id == project.id)
        .order_by(ReviewItem.created_at.asc())
    ).all()

    comments = [_serialize(item) for item in items if item.kind == "comment"]
    approvals = [_serialize(item) for item in items if item.kind == "approval"]
    return {
        "shared": True,
        "project": {
            "id": project.id,
            "name": project.name,
            "resolution_w": project.resolution_w,
            "resolution_h": project.resolution_h,
            "fps": project.fps,
        },
        "shots": [{
            "id": shot.id,
            "sequence_index": shot.sequence_index,
            "track_index": shot.track_index,
            "action": shot.action,
            "prompt": shot.prompt,
            "dialogue": shot.dialogue,
            "character": shot.character,
            "status": shot.status,
            "num_frames": shot.num_frames,
            "fps": shot.fps,
            "trim_in": shot.trim_in,
            "trim_out": shot.trim_out,
            "video_url": shot.video_url,
            "thumbnail_url": shot.thumbnail_url,
            "selected_take_id": shot.selected_take_id,
        } for shot in shots],
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


@router.post("/share/{token}/comments")
def add_share_comment(
    token: str,
    req: ShareCommentRequest,
    session: Session = Depends(get_session),
):
    link = _share_or_404(session, token)
    _check_shot(session, link.project_id, req.shot_id)
    body = (req.body or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail="Comment body is required")
    if req.frame < 0:
        raise HTTPException(status_code=400, detail="Frame must be zero or greater")

    item = ReviewItem(
        project_id=link.project_id,
        shot_id=req.shot_id,
        kind="comment",
        frame=req.frame,
        body=body[:4000],
        author=(req.author or "reviewer")[:80],
        status="open",
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return _serialize(item)


@router.post("/share/{token}/approvals")
def set_share_approval(
    token: str,
    req: ShareApprovalRequest,
    session: Session = Depends(get_session),
):
    link = _share_or_404(session, token)
    _check_shot(session, link.project_id, req.shot_id)

    item = session.exec(
        select(ReviewItem)
        .where(ReviewItem.project_id == link.project_id)
        .where(ReviewItem.shot_id == req.shot_id)
        .where(ReviewItem.kind == "approval")
    ).first()
    if item:
        item.status = req.status
        item.body = (req.note or "")[:4000]
        item.author = (req.author or "reviewer")[:80]
        item.updated_at = datetime.utcnow()
    else:
        item = ReviewItem(
            project_id=link.project_id,
            shot_id=req.shot_id,
            kind="approval",
            frame=0,
            body=(req.note or "")[:4000],
            author=(req.author or "reviewer")[:80],
            status=req.status,
        )
    session.add(item)
    session.commit()
    session.refresh(item)
    return _serialize(item)
