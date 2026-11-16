from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from sqlmodel import Session, select, SQLModel
from database import get_session, Project, Shot, Scene
from schemas import CreateProjectRequest, ProjectState, TimelineItem, RenderRequest, RenderSettings
import uuid
import os
import shutil
import logging
import json
from datetime import datetime
import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["projects"])


def _parse_json_field(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _shot_dict(shot: Shot) -> dict:
    return {
        "id": shot.id,
        "project_id": shot.project_id,
        "scene_id": shot.scene_id,
        "index": shot.index,
        "sequence_index": shot.sequence_index,
        "prompt": shot.prompt,
        "width": shot.width,
        "height": shot.height,
        "seed": shot.seed,
        "num_frames": shot.num_frames,
        "fps": shot.fps,
        "cfg_scale": shot.cfg_scale,
        "negative_prompt": shot.negative_prompt,
        "enhance_prompt": shot.enhance_prompt,
        "upscale": shot.upscale,
        "pipeline_override": shot.pipeline_override,
        "auto_continue": shot.auto_continue,
        "status": shot.status,
        "thumbnail_url": shot.thumbnail_url,
        "video_url": shot.video_url,
        "track_index": shot.track_index,
        "start_frame": shot.start_frame,
        "trim_in": shot.trim_in,
        "trim_out": shot.trim_out,
        "action": shot.action,
        "dialogue": shot.dialogue,
        "character": shot.character,
        "shot_type": shot.shot_type,
        "timeline": _parse_json_field(shot.timeline, []),
        "matched_elements": _parse_json_field(shot.matched_elements, None),
        "enhanced_prompt_result": shot.enhanced_prompt_result,
        "last_job_id": shot.last_job_id,
        "created_at": shot.created_at.isoformat() if shot.created_at else None,
    }

@router.post("/projects", response_model=ProjectState)
def create_project(req: CreateProjectRequest, session: Session = Depends(get_session)):
    project_id = str(uuid.uuid4())[:8]
    
    # Create directory structure
    project_dir = os.path.join(config.PROJECTS_DIR, project_id)
    os.makedirs(os.path.join(project_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "thumbnails"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "workspace"), exist_ok=True) # For intermediate files
    
    new_project = Project(
        id=project_id,
        name=req.name,
        resolution_w=req.resolution_w,
        resolution_h=req.resolution_h,
        fps=req.fps,
        seed=req.seed
    )
    session.add(new_project)
    session.commit()
    session.refresh(new_project)
    
    # Return empty state
    return ProjectState(
        id=new_project.id,
        name=new_project.name,
        shots=[],
        resolution_w=new_project.resolution_w,
        resolution_h=new_project.resolution_h,
        fps=new_project.fps,
        seed=new_project.seed,
        created_at=new_project.created_at.timestamp() if new_project.created_at else None,
        updated_at=new_project.updated_at.timestamp() if new_project.updated_at else None,
        revision=new_project.revision,
    )

@router.get("/projects", response_model=List[ProjectState])
def get_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project)).all()
    results = []
    for p in projects:
        # Get shot count or details? For list, maybe just summary, 
        # but ProjectState definition expects shots list.
        # We can optimize later, for now load shots.
        # Ensure we return valid shots structure
        shots_data = [] # We might need to load shots if we want to show them
        results.append(ProjectState(
            id=p.id,
            name=p.name,
            shots=[], # Optimization: Don't load all shots for index
            resolution_w=p.resolution_w,
            resolution_h=p.resolution_h,
            fps=p.fps,
            seed=p.seed,
            created_at=p.created_at.timestamp() if p.created_at else None,
            updated_at=p.updated_at.timestamp() if p.updated_at else None,
            revision=p.revision,
        ))
    return results

@router.get("/projects/{project_id}", response_model=ProjectState)
def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Load shots
    # Load scenes
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id).order_by(Scene.index)).all()
    scenes_data = []
    for sc in scenes:
        scenes_data.append({
            "id": sc.id,
            "index": sc.index,
            "name": sc.name,
            "script_content": sc.script_content
        })

    shots = session.exec(
        select(Shot)
        .where(Shot.project_id == project_id)
        .order_by(Shot.sequence_index.is_(None), Shot.sequence_index, Shot.created_at, Shot.id)
    ).all()

    return ProjectState(
        id=project.id,
        name=project.name,
        shots=[_shot_dict(s) for s in shots],
        scenes=scenes_data,
        resolution_w=project.resolution_w,
        resolution_h=project.resolution_h,
        fps=project.fps,
        seed=project.seed,
        created_at=project.created_at.timestamp() if project.created_at else None,
        updated_at=project.updated_at.timestamp() if project.updated_at else None,
        revision=project.revision,
    )

@router.delete("/projects/{project_id}")
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")
         
    # 1. Delete DB Records
    # Cascade delete shots? SQLModel might not handle cascade automatically unless configured
    shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    for s in shots:
        session.delete(s)
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    for sc in scenes:
        session.delete(sc)
    session.delete(project)
    session.commit()
    
    # 2. Delete Filesystem
    project_dir = os.path.join(config.PROJECTS_DIR, project_id)
    if os.path.exists(project_dir):
        try:
            shutil.rmtree(project_dir)
            logger.info(f"Deleted project directory: {project_dir}")
        except Exception as e:
            logger.error(f"Failed to delete project directory {project_dir}: {e}")
    
    return {"status": "deleted"}

@router.put("/projects/{project_id}")
async def save_project(project_id: str, state: ProjectState, session: Session = Depends(get_session)):
    if state.id != project_id:
        raise HTTPException(status_code=400, detail="Project ID mismatch")
    
    db_project = session.get(Project, project_id)
    if not db_project:
        # UPSERT logic: If project doesn't exist (e.g., cleared DB but frontend still has default UI state), recreate it.
        logger.info(f"Project {project_id} not found, recreating...")
        db_project = Project(
            id=project_id,
            name=state.name,
            resolution_w=state.resolution_w,
            resolution_h=state.resolution_h,
            fps=state.fps,
            seed=state.seed
        )
        session.add(db_project)
    elif state.revision is not None and state.revision != db_project.revision:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Project revision conflict: client has {state.revision}, "
                f"server has {db_project.revision}. Reload before saving."
            ),
        )

    db_project.name = state.name
    
    # Save settings
    db_project.resolution_w = state.resolution_w
    db_project.resolution_h = state.resolution_h
    db_project.fps = state.fps
    db_project.seed = state.seed
    
    db_project.updated_at = datetime.utcnow()
    db_project.revision = (db_project.revision or 0) + 1

    # Sync Shots (Smart Merge)
    # Sync Scenes
    existing_scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    existing_scene_map = {sc.id: sc for sc in existing_scenes}
    scene_payload_ids = set()
    scenes_to_add = []

    for sc_data in state.scenes:
        sc_id = sc_data.get("id")
        if not sc_id:
             sc_id = uuid.uuid4().hex
             sc_data["id"] = sc_id
        
        scene_payload_ids.add(sc_id)
        
        if sc_id in existing_scene_map:
             existing_scene = existing_scene_map[sc_id]
             existing_scene.name = sc_data["name"]
             existing_scene.index = sc_data.get("index", 0)
             existing_scene.script_content = sc_data.get("script_content")
             session.add(existing_scene)
        else:
             new_scene = Scene(
                 id=sc_id,
                 project_id=project_id,
                 name=sc_data["name"],
                 index=sc_data.get("index", 0),
                 script_content=sc_data.get("script_content")
             )
             scenes_to_add.append(new_scene)
             
    # Delete removed scenes
    for sc in existing_scenes:
        if sc.id not in scene_payload_ids:
            session.delete(sc)
            
    for sc in scenes_to_add:
        session.add(sc)


    # Sync Shots (Smart Merge)
    valid_keys = Shot.model_fields.keys()
    
    # 1. Fetch Key Existing Data to Preserve
    # Map Shot ID -> Existing Shot DB Object
    existing_shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    existing_map = {s.id: s for s in existing_shots}
    
    # Track IDs in the new payload to identify deletions
    payload_ids = set()
    
    new_shots_to_add = []
    
    for order, s_data in enumerate(state.shots):
        shot_id = s_data.get("id")
        if not shot_id:
            shot_id = uuid.uuid4().hex
            s_data["id"] = shot_id
        payload_ids.add(shot_id)

        # Clean input data. The payload array is the canonical story order.
        clean_data = {k: v for k, v in s_data.items() if k in valid_keys}
        clean_data["project_id"] = project_id
        clean_data["sequence_index"] = order
        if "timeline" in clean_data:
            clean_data["timeline"] = json.dumps(
                TimelineItem.normalize(clean_data.get("timeline"))
            )
        if "matched_elements" in clean_data:
            matched = _parse_json_field(clean_data.get("matched_elements"), None)
            clean_data["matched_elements"] = json.dumps(matched) if matched is not None else None

        if shot_id and shot_id in existing_map:
            # UPDATE Existing Shot
            existing_shot = existing_map[shot_id]

            for k, v in clean_data.items():
                if k in ["timeline", "matched_elements"]:
                    setattr(existing_shot, k, v)
                else:
                    setattr(existing_shot, k, v)

            session.add(existing_shot)
        else:
            # CREATE New Shot
            if not clean_data.get("id"):
                clean_data["id"] = uuid.uuid4().hex

            new_shots_to_add.append(Shot(**clean_data))
            
    # 2. DELETE output shots that are not in payload
    # (Unless we want to archive them? For now, standard sync means delete removed items)
    for s in existing_shots:
        if s.id not in payload_ids:
            session.delete(s)
            
    # 3. Add New
    for s in new_shots_to_add:
        session.add(s)
        
    session.add(db_project)
    session.commit()
    session.refresh(db_project)

    return {"status": "saved", "revision": db_project.revision}

@router.patch("/projects/{project_id}/settings")
async def update_project_settings(project_id: str, updates: dict, session: Session = Depends(get_session)):
    """Update project settings (resolution, fps, seed, name) after creation."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    allowed_fields = {"name", "resolution_w", "resolution_h", "fps", "seed"}
    applied = {}
    for key, value in updates.items():
        if key in allowed_fields and value is not None:
            setattr(project, key, value)
            applied[key] = value
    
    if not applied:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    session.add(project)
    session.commit()
    session.refresh(project)
    
    return {
        "status": "updated",
        "settings": {
            "name": project.name,
            "resolution_w": project.resolution_w,
            "resolution_h": project.resolution_h,
            "fps": project.fps,
            "seed": project.seed
        }
    }

async def _has_audio_stream(path: str) -> bool:
    import asyncio
    import subprocess
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return bool(stdout.strip())
    except Exception:
        return False


@router.post("/projects/{project_id}/render")
async def render_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    req: Optional[RenderRequest] = None,
    session: Session = Depends(get_session),
):
    """Start a durable render job for the project's computed timeline."""
    from database import Job

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    render_id = f"render_{uuid.uuid4().hex[:8]}"
    settings = (req.settings if req else RenderSettings()) or RenderSettings()
    track_states = (req.track_states if req else {}) or {}
    fps = settings.fps or project.fps
    width = settings.width or project.resolution_w
    height = settings.height or project.resolution_h

    job = Job(
        id=render_id,
        project_id=project_id,
        type="render",
        status="processing",
        progress=0,
        status_message="Initializing render...",
        params_json=json.dumps({
            "settings": settings.model_dump(),
            "track_states": track_states,
        }),
    )
    session.add(job)
    session.commit()

    background_tasks.add_task(
        _render_background,
        project_id, render_id, fps, width, height, settings, track_states,
    )
    return {"job_id": render_id, "status": "rendering"}


async def _render_background(
    project_id: str,
    render_id: str,
    fps: int,
    width: int,
    height: int,
    settings: RenderSettings,
    track_states: dict,
):
    """Build the filtergraph from computed layout, render, and stream progress."""
    import asyncio
    from events import event_manager
    from utils.render_utils import build_complex_filtergraph
    from file_utils import resolve_path
    from database import engine, Job
    from job_utils import active_jobs

    active_jobs[render_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "status_message": "Initializing render...",
        "project_id": project_id,
        "type": "render",
    }

    with Session(engine) as session:
        job = session.get(Job, render_id)
        if not job:
            logger.error(f"Render Job {render_id} missing in background thread.")
            active_jobs.pop(render_id, None)
            return

        project_dir = os.path.join(config.PROJECTS_DIR, project_id, "generated")
        os.makedirs(project_dir, exist_ok=True)
        output_path = os.path.join(project_dir, f"{render_id}.mp4")
        video_url = f"/projects/{project_id}/generated/{render_id}.mp4"

        async def emit_progress(pct: int, message: str, status: str = "processing"):
            pct = max(0, min(100, int(pct)))
            if render_id in active_jobs:
                active_jobs[render_id].update({
                    "progress": pct, "status": status if status != "processing" else "processing",
                    "status_message": message,
                })
            job.progress = pct
            job.status_message = message
            job.status = "processing" if status == "processing" else status
            if status == "completed":
                job.output_path = video_url
                job.completed_at = datetime.utcnow()
            elif status in ("failed", "cancelled"):
                job.error_message = message if status == "failed" else None
                job.completed_at = datetime.utcnow()
            session.add(job)
            session.commit()
            session.refresh(job)

            payload = {
                "job_id": render_id,
                "project_id": project_id,
                "progress": pct,
                "message": message,
                "status": status,
            }
            if status == "completed":
                payload["video_url"] = video_url
            elif status == "failed":
                payload["error"] = message
            await event_manager.broadcast(
                "render_progress" if status == "processing" else f"render_{status}",
                payload,
            )

        try:
            await emit_progress(2, "Gathering computed timeline...")

            all_shots = session.exec(
                select(Shot)
                .where(Shot.project_id == project_id)
                .order_by(Shot.sequence_index.is_(None), Shot.sequence_index, Shot.created_at, Shot.id)
            ).all()

            v1_clips, v2_clips, a1_clips = [], [], []
            v1_cursor = 0

            for shot in all_shots:
                if not shot.video_url:
                    continue
                resolved = resolve_path(shot.video_url)
                if not resolved or not os.path.exists(resolved):
                    logger.warning(f"Render skipping missing media: {shot.video_url}")
                    continue

                duration_frames = max(
                    0,
                    (shot.num_frames or 121) - (shot.trim_in or 0) - (shot.trim_out or 0),
                )
                if duration_frames <= 0:
                    continue

                if shot.track_index == 0:
                    start_frame = v1_cursor
                    v1_cursor += duration_frames
                else:
                    start_frame = shot.start_frame or 0

                clip = {
                    "path": resolved,
                    "start_frame": start_frame,
                    "duration_frames": duration_frames,
                    "trim_in": shot.trim_in or 0,
                    "gain": 1.0,
                    "muted": False,
                    "hidden": False,
                    "has_audio": await _has_audio_stream(resolved),
                }
                if shot.track_index == 0:
                    v1_clips.append(clip)
                elif shot.track_index == 1:
                    v2_clips.append(clip)
                elif shot.track_index == 2:
                    a1_clips.append(clip)

            if not v1_clips and not v2_clips and not a1_clips:
                raise ValueError("No valid video or audio clips found to render.")

            await emit_progress(5, "Compiling FFmpeg filtergraph...")
            cmd = build_complex_filtergraph(
                project_fps=fps,
                project_width=width,
                project_height=height,
                v1_shots=v1_clips,
                v2_shots=v2_clips,
                a1_shots=a1_clips,
                output_path=output_path,
                track_states=track_states,
                crf=settings.crf,
                preset=settings.preset,
                audio_bitrate=settings.audio_bitrate,
            )

            logger.info("Executing render FFmpeg: %s", " ".join(cmd))
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stderr_task = asyncio.create_task(process.stderr.read())

            duration_sec = max(1.0, (
                max((c["start_frame"] + c["duration_frames"]) for c in (v1_clips + v2_clips + a1_clips))
                / max(1, fps)
            ))

            cancelled = False
            while True:
                if active_jobs.get(render_id, {}).get("cancelled"):
                    cancelled = True
                    process.terminate()
                    break
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "ignore").strip()
                if text.startswith("out_time_us=") or text.startswith("out_time_ms="):
                    raw = text.split("=", 1)[1]
                    try:
                        # out_time_us is microseconds; out_time_ms is misleadingly
                        # microseconds in recent FFmpeg as well.
                        seconds = int(raw) / 1_000_000
                        pct = 5 + min(94, int((seconds / duration_sec) * 94))
                        await emit_progress(pct, f"Rendering {seconds:.1f}s / {duration_sec:.1f}s")
                    except ValueError:
                        pass

            returncode = await process.wait()
            stderr = (await stderr_task).decode("utf-8", "ignore")
            if cancelled:
                await emit_progress(0, "Render cancelled", status="cancelled")
                await event_manager.broadcast("cancelled", {"job_id": render_id})
                return
            if returncode != 0:
                raise RuntimeError(f"FFmpeg render failed: {stderr[-4000:]}")

            await emit_progress(100, "Render complete!", status="completed")
            logger.info(f"Render completed: {output_path}")

        except Exception as e:
            logger.error(f"Render failed: {e}")
            await emit_progress(0, str(e), status="failed")
        finally:
            active_jobs.pop(render_id, None)


@router.get("/projects/{project_id}/renders")
async def get_project_renders(project_id: str, session: Session = Depends(get_session)):
    """Fetch history of completed project renders."""
    from database import Job
    statement = select(Job).where(Job.project_id == project_id, Job.type == "render").order_by(Job.created_at.desc())
    renders = session.exec(statement).all()
    return renders


@router.get("/projects/{project_id}/render/{render_id}/download")
async def download_render(project_id: str, render_id: str):
    """Serve the rendered video file as a download."""
    from fastapi.responses import FileResponse
    
    file_path = os.path.join(config.PROJECTS_DIR, project_id, "generated", f"{render_id}.mp4")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Render not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=f"{render_id}.mp4"
    )

@router.get("/projects/{project_id}/images")
async def get_project_images(project_id: str, session: Session = Depends(get_session)):
    """List all generated images for a project."""
    from database import Asset
    statement = select(Asset).where(Asset.project_id == project_id, Asset.type == "image").order_by(Asset.created_at.desc())
    results = session.exec(statement).all()
    return results


class SplitShotRequest(SQLModel):
    split_frame: int # Local frame index in the shot

@router.post("/shots/{shot_id}/split")
async def split_shot(shot_id: str, req: SplitShotRequest, session: Session = Depends(get_session)):
    """
    Split a shot into two at the given local frame index.
    Adjusts trim_out of original and creates new shot with trim_in.
    """
    original_shot = session.get(Shot, shot_id)
    if not original_shot:
        raise HTTPException(status_code=404, detail="Shot not found")
        
    project = session.get(Project, original_shot.project_id)
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")

    # 1. Validate Split Point
    # Must be > 0 and < (num_frames - trim_in - trim_out)
    current_duration = original_shot.num_frames - (original_shot.trim_in or 0) - (original_shot.trim_out or 0)
    
    if req.split_frame <= 0 or req.split_frame >= current_duration:
        raise HTTPException(status_code=400, detail="Split point out of bounds")
        
    # Absolute frame within the source video
    absolute_split_frame = (original_shot.trim_in or 0) + req.split_frame
    
    # 2. Update Original Shot (Trim Out)
    # The new trim_out should be such that shot ends at absolute_split_frame
    # num_frames - trim_out_new = absolute_split_frame
    # trim_out_new = num_frames - absolute_split_frame
    
    old_trim_out = original_shot.trim_out or 0
    new_trim_out = original_shot.num_frames - absolute_split_frame
    
    original_shot.trim_out = new_trim_out
    session.add(original_shot)
    
    # 3. Shift the project-wide story order and the scene-local index so the
    # new shot lands immediately after the original, not at the end.
    new_sequence_index = None
    if original_shot.sequence_index is not None:
        new_sequence_index = original_shot.sequence_index + 1
        for other in session.exec(
            select(Shot)
            .where(Shot.project_id == original_shot.project_id)
            .where(Shot.sequence_index >= new_sequence_index)
        ).all():
            other.sequence_index = (other.sequence_index or 0) + 1
            session.add(other)

    new_scene_index = None
    if original_shot.index is not None and original_shot.scene_id:
        new_scene_index = original_shot.index + 1
        for other in session.exec(
            select(Shot)
            .where(Shot.scene_id == original_shot.scene_id)
            .where(Shot.index >= new_scene_index)
        ).all():
            other.index = (other.index or 0) + 1
            session.add(other)

    # 4. Create New Shot (Copy of Original)
    new_shot_data = original_shot.model_dump(exclude={"id", "created_at"})
    new_shot = Shot(**new_shot_data)
    new_shot.id = uuid.uuid4().hex
    new_shot.created_at = datetime.utcnow()
    new_shot.sequence_index = new_sequence_index
    new_shot.index = new_scene_index
    new_shot.trim_in = absolute_split_frame
    new_shot.trim_out = old_trim_out

    # 5. Free tracks keep absolute position; V1 is magnetic from sequence_index.
    if original_shot.track_index != 0:
        new_shot.start_frame = (original_shot.start_frame or 0) + req.split_frame

    session.add(new_shot)
    session.commit()
    session.refresh(original_shot)
    session.refresh(new_shot)

    def _as_dict(shot: Shot) -> dict:
        data = shot.model_dump()
        for key in ("timeline", "matched_elements"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value) if value else ([] if key == "timeline" else None)
                except json.JSONDecodeError:
                    data[key] = [] if key == "timeline" else None
        return data

    return {
        "original_shot": _as_dict(original_shot),
        "new_shot": _as_dict(new_shot),
    }


# ── LLM Settings ──────────────────────────────────────────────────

@router.get("/settings/llm")
def get_llm_settings():
    """Return current LLM configuration."""
    return {
        "provider": config.LLM_PROVIDER,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_keep_alive": config.OLLAMA_KEEP_ALIVE,
    }


@router.patch("/settings/llm")
def update_llm_settings(body: dict):
    """Update LLM settings at runtime."""
    if "provider" in body:
        if body["provider"] not in ("gemma", "ollama"):
            raise HTTPException(400, f"Invalid provider: {body['provider']}. Must be 'gemma' or 'ollama'.")
        config.LLM_PROVIDER = body["provider"]
    if "ollama_model" in body:
        config.OLLAMA_MODEL = body["ollama_model"]
    if "ollama_base_url" in body:
        config.OLLAMA_BASE_URL = body["ollama_base_url"]
    if "ollama_keep_alive" in body:
        config.OLLAMA_KEEP_ALIVE = body["ollama_keep_alive"]
    
    logger.info(f"LLM settings updated: provider={config.LLM_PROVIDER}, model={config.OLLAMA_MODEL}, keep_alive={config.OLLAMA_KEEP_ALIVE}")
    return get_llm_settings()


# Vision model families — these can accept images via Ollama's multimodal API
VISION_FAMILIES = {"mllama", "clip", "llava", "bakllava", "moondream"}

@router.get("/settings/llm/models")
def list_ollama_models():
    """Query Ollama for available models, tagging vision-capable ones."""
    import requests
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        
        # Detect which models support vision (image input)
        result = []
        for m in models:
            families = set(m.get("details", {}).get("families") or [])
            is_vision = bool(families & VISION_FAMILIES)
            # Skip embedding-only models (bert, nomic-bert)
            family = m.get("details", {}).get("family", "")
            if family in ("bert", "nomic-bert"):
                continue
            result.append({
                "name": m["name"],
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
                "is_vision": is_vision,
                "parameter_size": m.get("details", {}).get("parameter_size"),
            })
        
        return {"models": result}
    except requests.ConnectionError:
        raise HTTPException(503, f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}")
    except Exception as e:
        raise HTTPException(500, str(e))
