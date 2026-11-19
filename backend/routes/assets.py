from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
import asyncio
import os
import re
import shutil
import subprocess
import uuid
import config
from file_utils import get_base_dir, resolve_path
from sqlmodel import Session
from database import get_session
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(tags=["assets"])


def _project_asset_dir(project_id: str, preferred: Optional[str] = None) -> str:
    """Choose a destination inside the project; never a legacy global folder."""
    root = os.path.realpath(os.path.join(config.PROJECTS_DIR, project_id))
    if preferred:
        real = os.path.realpath(preferred)
        try:
            if os.path.commonpath([root, real]) == root and os.path.isdir(real):
                return real
        except ValueError:
            pass
    return os.path.join(root, "assets", "general")


def _asset_url(project_id: Optional[str], file_path: str) -> str:
    if project_id:
        root = os.path.realpath(os.path.join(config.PROJECTS_DIR, project_id))
        real = os.path.realpath(file_path)
        try:
            if os.path.commonpath([root, real]) == root:
                rel = os.path.relpath(real, root).replace(os.sep, "/")
                return f"/projects/{project_id}/{rel}"
        except ValueError:
            pass
    return f"/uploads/{os.path.basename(file_path)}"


def _other_asset_refs(session: Session, asset_id: str, refs: set[str]) -> set[str]:
    """Paths/URLs still used by another Asset row (do not purge those)."""
    from database import Asset
    from sqlalchemy import or_
    from sqlmodel import select

    conditions = []
    for ref in refs:
        if not ref:
            continue
        conditions.extend([Asset.path == ref, Asset.url == ref, Asset.thumbnail_url == ref])
    if not conditions:
        return set()
    rows = session.exec(select(Asset).where(Asset.id != asset_id, or_(*conditions))).all()
    shared: set[str] = set()
    for row in rows:
        if row.path:
            shared.add(row.path)
        if row.url:
            shared.add(row.url)
        if row.thumbnail_url:
            shared.add(row.thumbnail_url)
    return shared

def _thumbnail_target(project_id: Optional[str], filename: str) -> tuple[str, str]:
    """Return (filesystem path, web URL) for an uploaded video poster."""
    stem = Path(filename).stem
    thumb_name = f"{stem}_thumb.jpg"
    if project_id:
        folder = os.path.join(config.PROJECTS_DIR, project_id, "thumbnails")
        url = f"/projects/{project_id}/thumbnails/{thumb_name}"
    else:
        folder = os.path.join(UPLOAD_DIR, "thumbnails")
        url = f"/uploads/thumbnails/{thumb_name}"
    return os.path.join(folder, thumb_name), url


def _make_video_thumbnail(video_path: str, thumb_path: str) -> bool:
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    for offset in ("0.5", "0"):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", offset, "-i", video_path,
                    "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
                    "-q:v", "4", thumb_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
                return True
        except (subprocess.CalledProcessError, OSError):
            continue
    try:
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    except OSError:
        pass
    return False

# Global upload dir (legacy)
UPLOAD_DIR = os.path.join(get_base_dir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    type: Optional[str] = Form("general"), # general, element, reference, lut
    session: Session = Depends(get_session)
):
    upload_target_dir = UPLOAD_DIR

    # Project-scoped uploads
    if project_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
            raise HTTPException(status_code=400, detail="Invalid project id")
        project_dir = os.path.join(config.PROJECTS_DIR, project_id)
        if not os.path.exists(project_dir):
            raise HTTPException(status_code=404, detail="Project not found")

        if type == "element":
            upload_target_dir = os.path.join(project_dir, "assets", "elements")
        elif type == "reference":
            upload_target_dir = os.path.join(project_dir, "assets", "references")
        elif type == "lut":
            upload_target_dir = os.path.join(project_dir, "assets", "lut")
        else:
             upload_target_dir = os.path.join(project_dir, "assets", "general")

        os.makedirs(upload_target_dir, exist_ok=True)

    # Never trust a client filename. Keep only the basename and enforce an
    # extension allowlist before anything touches the filesystem.
    original_name = Path(file.filename or "upload").name
    original_name = re.sub(r"[\x00-\x1f]", "", original_name).strip()
    if not original_name or original_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    extension = Path(original_name).suffix.lower()
    if extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{extension or 'none'}'",
        )

    stem = Path(original_name).stem[:120]
    filename = f"{uuid.uuid4().hex[:8]}_{stem}{extension}"
    file_path = os.path.join(upload_target_dir, filename)
    temp_path = f"{file_path}.part"
    max_bytes = config.MAX_UPLOAD_BYTES
    written = 0
    try:
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {config.MAX_UPLOAD_MB} MB limit",
                    )
                buffer.write(chunk)
        os.replace(temp_path, file_path)
    except Exception:
        for path in (temp_path, file_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise
    finally:
        file.file.close()

    # Create DB Record
    from database import Asset
    from datetime import datetime, timezone

    # Calculate access URL
    if project_id:
        folder_name = "general"
        if type == "element": folder_name = "elements"
        if type == "reference": folder_name = "references"
        if type == "lut": folder_name = "lut"
        access_url = f"/projects/{project_id}/assets/{folder_name}/{filename}"
    else:
        access_url = f"/uploads/{filename}"

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    caption_extensions = {".srt", ".vtt"}
    text_extensions = {".json", ".txt", ".md"}
    lut_extensions = {".cube", ".3dl", ".dat", ".m3d"}
    asset_type = (
        "image" if extension in image_extensions
        else "audio" if extension in audio_extensions
        else "caption" if extension in caption_extensions
        else "text" if extension in text_extensions
        else "lut" if extension in lut_extensions
        else "video"
    )
    thumbnail_url = None
    if asset_type == "video":
        thumb_path, thumb_url = _thumbnail_target(project_id, filename)
        if _make_video_thumbnail(file_path, thumb_path):
            thumbnail_url = thumb_url

    new_asset = Asset(
        id=uuid.uuid4().hex,
        project_id=project_id,
        type=asset_type,
        url=access_url,
        path=file_path,
        filename=filename,
        thumbnail_url=thumbnail_url,
        created_at=datetime.now(timezone.utc),
        meta_json=None,
    )
    session.add(new_asset)
    session.commit()

    return {
        "url": access_url,
        "asset_id": new_asset.id,
        "type": new_asset.type,
        "filename": new_asset.filename,
        "thumbnail": new_asset.thumbnail_url,
        "size_bytes": written,
    }

@router.post("/shot/{job_id}/last-frame")
async def get_shot_last_frame(job_id: str, session: Session = Depends(get_session)):
    """
    Returns the path/url to the last frame of a generated video.
    Used for 'Extend' functionality.
    """
    import subprocess
    from database import get_session, Job

    # Look up the job in DB to get project-scoped output path
    job = session.get(Job, job_id)

    if not job or not job.output_path:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or has no output")

    # Require project-scoped paths
    if not job.output_path.startswith("/projects/"):
        raise HTTPException(status_code=400, detail=f"Job {job_id} uses legacy path format.")

    # Project-scoped: /projects/{id}/generated/filename.mp4
    # We need to resolve to FS path. get_base_dir() returns backend root.
    # config.PROJECTS_DIR is absolute.
    # job.output_path looks like /projects/ID/generated/foo.mp4

    rel_path = job.output_path.removeprefix("/projects/")
    video_path = os.path.join(config.PROJECTS_DIR, rel_path)

    # Check if it's an image (single frame generation)
    if job.output_path.endswith(".jpg") or job.output_path.endswith(".png"):
        if os.path.exists(video_path):
            return {"url": job.output_path, "path": video_path}
        raise HTTPException(status_code=404, detail=f"Image not found: {job.output_path}")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    try:
        # Extract last frame to project workspace
        out_name = f"{job_id}_last.jpg"

        # Determine output directory (same as video)
        out_dir = os.path.dirname(video_path)
        out_path = os.path.join(out_dir, out_name)

        # Extract last frame using ffmpeg
        await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-sseof", "-1.0", "-i", video_path,
             "-q:v", "2", "-update", "1", out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Build URL based on original path pattern
        parts = job.output_path.split("/")
        project_id = parts[2]
        out_url = f"/projects/{project_id}/generated/{out_name}"

        return {"url": out_url, "path": out_path}
    except Exception as e:
        # logger.error(f"Failed to extract last frame from {video_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploads")
async def list_uploads(
    project_id: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """
    Returns a combined list of Assets (Uploaded) and Completed Jobs (Generated)
    from the SQLite database, deduplicated by URL.

    Pass `project_id` to scope the result to one project. Omitting it keeps the
    legacy global view for callers that explicitly want every project.
    """
    from database import Asset, Job
    from sqlmodel import select

    files = []
    seen_urls = set()  # Track URLs to avoid duplicates

    # 1. Get Assets (Uploaded + Generated Images) - these take priority
    asset_query = select(Asset)
    if project_id:
        asset_query = asset_query.where(Asset.project_id == project_id)
    assets = session.exec(asset_query.order_by(Asset.created_at.desc())).all()
    for a in assets:
        if a.url in seen_urls:
            continue
        seen_urls.add(a.url)
        thumb = a.thumbnail_url
        source_path = a.path if a.path and os.path.exists(a.path) else resolve_path(a.url)
        if a.type == "video" and not thumb and source_path and os.path.exists(source_path):
            thumb_path, thumb_url = _thumbnail_target(a.project_id, a.filename)
            if await asyncio.to_thread(_make_video_thumbnail, source_path, thumb_path):
                a.thumbnail_url = thumb_url
                session.add(a)
                session.commit()
                thumb = thumb_url
        files.append({
            "id": a.id,
            "url": a.url,
            "path": a.path,
            "type": a.type,
            "filename": a.filename,
            "thumbnail": thumb,
            "missing": not bool(source_path and os.path.exists(source_path)),
            "created_at": a.created_at.timestamp()
        })

    # 2. Get Completed Jobs (Generated Videos) - skip if URL already seen
    job_query = select(Job).where(Job.status == "completed")
    if project_id:
        job_query = job_query.where(Job.project_id == project_id)
    jobs = session.exec(job_query.order_by(Job.created_at.desc())).all()


    for j in jobs:
        if not j.output_path: continue

        filename = os.path.basename(j.output_path)

        # Only serve project-scoped paths
        if j.output_path.startswith("/projects/"):
            video_url = j.output_path
        elif "/projects/" in j.output_path:
            # Handle absolute filesystem paths like PROJECTS_DIR/...
            if j.output_path.startswith(config.PROJECTS_DIR):
                rel = os.path.relpath(j.output_path, os.path.dirname(config.PROJECTS_DIR))
                video_url = f"/{rel}"
            else:
                # Skip - cannot determine correct URL
                continue
        else:
            # Skip legacy /generated/ paths - no longer supported (unless fallback needed)
            continue

        # Skip if already seen (e.g., image already in Assets)
        if video_url in seen_urls:
            continue
        seen_urls.add(video_url)

        # THUMBNAIL LOGIC
        thumb_url = None
        if j.thumbnail_path:
            thumb_url = j.thumbnail_path
        else:
            # Fallback: check filesystem (only for project paths)
            if filename.endswith(".mp4"):
                if "/projects/" in video_url:
                    try:
                        # video: /projects/ID/generated/video.mp4
                        # thumb: /projects/ID/thumbnails/video_thumb.jpg
                        parts = video_url.split("/")
                        if len(parts) >= 5 and parts[3] == "generated":
                            parts[3] = "thumbnails"
                            parts[-1] = parts[-1].replace(".mp4", "_thumb.jpg")
                            # Assume the derived thumbnail exists when the DB
                            # entry does; a missing file simply falls back to no image.
                            thumb_url = "/".join(parts)
                    except: pass

        job_path = resolve_path(j.output_path)
        files.append({
            "id": j.id,
            "url": video_url,
            "path": j.output_path,
            "type": "video" if filename.endswith(".mp4") else "image",
            "filename": filename,
            "thumbnail": thumb_url,
            "missing": not bool(job_path and os.path.exists(job_path)),
            "created_at": j.created_at.timestamp()
        })

    # Sort combined by time
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return files

@router.post("/assets/{asset_id}/replace")
def replace_asset_file(
    asset_id: str,
    file: UploadFile = File(...),
    purge_old: bool = Form(False),
    session: Session = Depends(get_session),
):
    """Relink a missing Library asset by replacing its file and URL.

    The asset id is stable, so shots that point at the old URL are rewritten to
    the new one. Nothing is deleted implicitly: a caller that wants the record
    gone still uses DELETE.
    """
    from database import Asset, ComfyWorkflow, Job, Project, Shot
    from datetime import datetime, timezone
    from sqlmodel import select

    asset = session.get(Asset, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    old_url = asset.url
    old_path = asset.path
    old_thumbnail = asset.thumbnail_url
    old_type = asset.type
    jobs = session.exec(
        select(Job).where(
            (Job.output_path == old_url) | (Job.thumbnail_path == old_url)
        )
    ).all()

    original_name = Path(file.filename or "upload").name
    original_name = re.sub(r"[\x00-\x1f]", "", original_name).strip()
    if not original_name or original_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    extension = Path(original_name).suffix.lower()
    if extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{extension or 'none'}'")

    folder = (
        _project_asset_dir(asset.project_id, os.path.dirname(asset.path) if asset.path else None)
        if asset.project_id else UPLOAD_DIR
    )
    os.makedirs(folder, exist_ok=True)
    stem = Path(original_name).stem[:120]
    filename = f"{uuid.uuid4().hex[:8]}_{stem}{extension}"
    file_path = os.path.join(folder, filename)
    temp_path = f"{file_path}.part"
    written = 0
    try:
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {config.MAX_UPLOAD_MB} MB limit")
                buffer.write(chunk)
        os.replace(temp_path, file_path)
    except Exception:
        for path in (temp_path, file_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise
    finally:
        file.file.close()

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    caption_extensions = {".srt", ".vtt"}
    text_extensions = {".json", ".txt", ".md"}
    lut_extensions = {".cube", ".3dl", ".dat", ".m3d"}
    new_type = (
        "image" if extension in image_extensions
        else "audio" if extension in audio_extensions
        else "caption" if extension in caption_extensions
        else "text" if extension in text_extensions
        else "lut" if extension in lut_extensions
        else "video"
    )
    if jobs and new_type != old_type:
        for path in (temp_path, file_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise HTTPException(
            status_code=409,
            detail=f"This asset backs {len(jobs)} take(s); replace it with another {old_type} file",
        )
    asset.type = new_type
    asset.url = _asset_url(asset.project_id, file_path)
    asset.path = file_path
    asset.filename = filename
    # A replacement invalidates prompt/seed/reference provenance.
    asset.meta_json = None
    # Preserve the original import date; relinking is not a re-import.
    if asset.type == "video":
        thumb_path, thumb_url = _thumbnail_target(asset.project_id, filename)
        asset.thumbnail_url = thumb_url if _make_video_thumbnail(file_path, thumb_path) else None
    else:
        asset.thumbnail_url = None

    if old_url and old_url != asset.url:
        old_refs = [ref for ref in {old_url, old_thumbnail} if ref]
        shots = session.exec(
            select(Shot).where(
                (Shot.video_url.in_(old_refs)) | (Shot.thumbnail_url.in_(old_refs))
            )
        ).all()
        for shot in shots:
            if shot.video_url in old_refs:
                shot.video_url = asset.url
            if shot.thumbnail_url in old_refs:
                shot.thumbnail_url = asset.thumbnail_url or asset.url
            session.add(shot)

        for job in jobs:
            if job.output_path in {old_url, old_thumbnail}:
                job.output_path = asset.url
            if job.thumbnail_path in {old_url, old_thumbnail}:
                job.thumbnail_path = asset.thumbnail_url
            session.add(job)

        projects = session.exec(select(Project).where(Project.render_lut == old_url)).all()
        for project in projects:
            project.render_lut = asset.url
            project.revision = (project.revision or 0) + 1
            project.updated_at = datetime.utcnow()
            session.add(project)

        workflows = session.exec(
            select(ComfyWorkflow).where(ComfyWorkflow.project_id == asset.project_id)
        ).all()
        for workflow in workflows:
            changed = False
            for field in ("graph_json", "bindings_json"):
                value = getattr(workflow, field) or ""
                if old_url in value:
                    setattr(workflow, field, value.replace(old_url, asset.url))
                    changed = True
            if changed:
                workflow.revision = (workflow.revision or 0) + 1
                workflow.updated_at = datetime.utcnow()
                session.add(workflow)

    removed_paths: list[str] = []
    shared_paths: set[str] = set()
    if purge_old:
        refs = {ref for ref in {old_url, old_path, old_thumbnail, resolve_path(old_thumbnail) if old_thumbnail else None} if ref}
        shared_paths = _other_asset_refs(session, asset.id, refs)
        candidates = {old_path, resolve_path(old_thumbnail) if old_thumbnail else None}
        for candidate in candidates:
            if not candidate or candidate == file_path or not os.path.exists(candidate):
                continue
            if os.path.realpath(candidate) in {os.path.realpath(path) for path in shared_paths}:
                continue
            try:
                os.remove(candidate)
                removed_paths.append(candidate)
            except OSError:
                pass

    removed_old = bool(old_path and old_path in removed_paths)
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return {
        "id": asset.id,
        "url": asset.url,
        "path": asset.path,
        "filename": asset.filename,
        "type": asset.type,
        "thumbnail": asset.thumbnail_url,
        "missing": False,
        "created_at": asset.created_at.timestamp(),
        "old_url": old_url,
        "old_path": old_path,
        "old_thumbnail": old_thumbnail,
        "old_removed": removed_old,
        "removed_paths": removed_paths,
        "old_shared": bool(shared_paths),
    }


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str, session: Session = Depends(get_session)):
    from database import Asset, Job, Shot
    from sqlmodel import select

    # 1. Try finding Asset by ID (Primary Key)
    asset = session.get(Asset, asset_id)
    if asset:
        target_url = asset.url
        session.delete(asset)

        # Clear any shot that points at this asset, including its poster.
        if target_url:
            shots = session.exec(
                select(Shot).where(
                    (Shot.video_url == target_url) | (Shot.thumbnail_url == target_url)
                )
            ).all()
            for shot in shots:
                if shot.video_url == target_url:
                    shot.video_url = None
                    shot.selected_take_id = None
                if shot.thumbnail_url == target_url:
                    shot.thumbnail_url = None
                session.add(shot)

        # Also clean up any Job pointing to this asset
        if target_url:
             # Try exact URL match
             jobs = session.exec(select(Job).where(Job.output_path == target_url)).all()

             # Fallback: Try match by filename (if URL match missed for some reason)
             if not jobs and asset.filename:
                 # Check if output_path ends with the filename
                 # SQLModel/SQLAlchemy 'endswith' or 'like'
                 jobs = session.exec(select(Job).where(Job.output_path.like(f"%/{asset.filename}"))).all()

             for j in jobs:
                 j.output_path = None
                 j.status = "deleted"
                 session.add(j)

        session.commit()

        # Delete the asset file and its poster, if present.
        for candidate in (asset.path, resolve_path(asset.thumbnail_url) if asset.thumbnail_url else None):
            if candidate and os.path.exists(candidate):
                try:
                    os.remove(candidate)
                except Exception as e:
                    logger.error(f"Failed to delete asset file {candidate}: {e}")

        return {"status": "deleted", "id": asset_id}

    # 2. Try finding Job (if ID passed was a Job ID)
    job = session.get(Job, asset_id)
    if job:
        # Resolve absolute path from URL
        file_path = job.output_path
        target_path = job.output_path # Keep for Asset lookup

        if file_path and file_path.startswith("/projects/"):
            rel = file_path[len("/projects/"):]
            file_path = os.path.join(config.PROJECTS_DIR, rel)

        if file_path and os.path.exists(file_path):
             if os.path.isdir(file_path):
                 shutil.rmtree(file_path)
             else:
                 try: os.remove(file_path)
                 except: pass

        # Clean up corresponding Asset if any
        if target_path:
             assets = session.exec(select(Asset).where(Asset.url == target_path)).all()
             # Or check path? Asset.path is absolute/resolved. Asset.url is web.
             # Job.output_path for images is usually the web URL in this codebase version?
             # Let's check Job creation in image.py:
             # update_job_db(..., output_path=web_url...)
             # So Job stores URL. Asset stores URL and Path.
             # So we match Job.output_path (URL) to Asset.url.
             for a in assets:
                 session.delete(a)

        job.output_path = None # Clear path
        job.status = "deleted"
        session.add(job)
        session.commit()
        return {"status": "deleted", "id": asset_id}

    # 3. Fallback: Try filename match (Legacy)
    # If the ID passed doesn't match a record, maybe it was a filename?
    # Only try this if it looks like a filename (has dot)
    if "." in asset_id:
        return await delete_upload(asset_id, session)

    raise HTTPException(status_code=404, detail="Asset not found")

@router.delete("/upload/{filename}")
async def delete_upload(filename: str, session: Session = Depends(get_session)):

    from database import Asset, Job
    from sqlmodel import select

    # Try finding Asset
    asset = session.exec(select(Asset).where(Asset.filename == filename)).first()
    if asset:
        session.delete(asset)
        session.commit()
        # Delete file
        if os.path.exists(asset.path):
             try:
                 os.remove(asset.path)
             except: pass
        return {"status": "deleted"}

    # Try finding Job (id usually matches filename base)
    job_id = filename.split(".")[0]
    job = session.get(Job, job_id)
    if job:
        # Resolve absolute path from URL
        file_path = job.output_path
        if file_path and file_path.startswith("/projects/"):
            rel = file_path.removeprefix("/projects/")
            # Safer slice
            rel = file_path[len("/projects/"):]
            file_path = os.path.join(config.PROJECTS_DIR, rel)

        if file_path and os.path.exists(file_path):
             if os.path.isdir(file_path):
                 shutil.rmtree(file_path)
             else:
                 try: os.remove(file_path)
                 except: pass

        job.output_path = None # Clear path
        job.status = "deleted"
        session.add(job)
        session.commit()
        return {"status": "deleted"}

    raise HTTPException(status_code=404, detail="File not found")
