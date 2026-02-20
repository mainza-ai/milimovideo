from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
from sqlmodel import Session, select, SQLModel
from database import get_session, Project, Shot, Scene
from schemas import CreateProjectRequest, ProjectState
import uuid
import os
import shutil
import logging
import json
import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["projects"])

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
        updated_at=new_project.updated_at.timestamp() if new_project.updated_at else None
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
            updated_at=p.updated_at.timestamp() if p.updated_at else None
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

    shots = session.exec(select(Shot).where(Shot.project_id == project_id).order_by(Shot.created_at)).all()
    
    shots_data = []
    for s in shots:
        shots_data.append({
            "id": s.id,
            "project_id": s.project_id,
            "prompt": s.prompt,
            "width": s.width,
            "height": s.height, 
            "seed": s.seed,
            "num_frames": s.num_frames,
            "fps": s.fps,
            "cfg_scale": s.cfg_scale,
            "negative_prompt": s.negative_prompt,
            "enhance_prompt": s.enhance_prompt,
            "upscale": s.upscale,
            "pipeline_override": s.pipeline_override,
            "auto_continue": s.auto_continue,
            "status": s.status,
            "thumbnail_url": s.thumbnail_url,
            "video_url": s.video_url,
            "track_index": s.track_index,
            "start_frame": s.start_frame,
            "trim_in": s.trim_in,
            "trim_out": s.trim_out,
            "scene_id": s.scene_id,
            "action": s.action,
            "dialogue": s.dialogue,
            "character": s.character,
            "shot_type": s.shot_type,
            "matched_elements": s.matched_elements,
            "enhanced_prompt_result": s.enhanced_prompt_result,
            "last_job_id": s.last_job_id,
            "created_at": s.created_at.isoformat() if s.created_at else None
        })
        
    return ProjectState(
        id=project.id,
        name=project.name,
        shots=shots_data,
        scenes=scenes_data,
        resolution_w=project.resolution_w,
        resolution_h=project.resolution_h,
        fps=project.fps,
        seed=project.seed,
        created_at=project.created_at.timestamp() if project.created_at else None,
        updated_at=project.updated_at.timestamp() if project.updated_at else None
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
    
    db_project.name = state.name
    
    # Save settings
    db_project.resolution_w = state.resolution_w
    db_project.resolution_h = state.resolution_h
    db_project.fps = state.fps
    db_project.seed = state.seed
    
    # db_project.updated_at = datetime.now(timezone.utc) # Handled by SQLModel typically or trigger
    
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
    
    for s_data in state.shots:
        shot_id = s_data.get("id")
        payload_ids.add(shot_id)
        
        # Clean input data
        clean_data = {k: v for k, v in s_data.items() if k in valid_keys}
        clean_data["project_id"] = project_id
        
        if shot_id and shot_id in existing_map:
            # UPDATE Existing Shot
            existing_shot = existing_map[shot_id]
            
            # Update fields from payload
            for k, v in clean_data.items():
                if k in ["timeline", "matched_elements"] and isinstance(v, (list, dict)):
                    setattr(existing_shot, k, json.dumps(v))
                else:
                    setattr(existing_shot, k, v)
                
            session.add(existing_shot)
        else:
            # CREATE New Shot
            # Ensure ID is set
            if not clean_data.get("id"):
                clean_data["id"] = uuid.uuid4().hex
            
            # Serialize JSON fields for creation
            for field in ["timeline", "matched_elements"]:
                if field in clean_data and isinstance(clean_data[field], (list, dict)):
                    clean_data[field] = json.dumps(clean_data[field])

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
        
    return {"status": "saved"}

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

@router.patch("/shots/{shot_id}")
async def update_shot_partial(shot_id: str, updates: dict, session: Session = Depends(get_session)):
    """
    Granular update for a single shot. 
    Useful for timeline drag-and-drop or text edits without full project save.
    """
    shot = session.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
        
    # Allowed fields to patch
    # We allow patching almost anything, but be careful with IDs.
    for k, v in updates.items():
        if k == "id" or k == "project_id": continue # Don't allow changing PK/FK via patch
        if hasattr(shot, k):
            setattr(shot, k, v)
            
    session.add(shot)
    session.commit()
    session.refresh(shot)
    return shot

@router.post("/projects/{project_id}/render")
async def render_project(project_id: str, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """
    Kicks off an async render job that stitches V1 shots + A1 audio into a final MP4.
    Progress is sent via SSE events: render_progress, render_complete, render_failed.
    Returns immediately with { job_id, status: "rendering" }.
    """
    from database import Job, Asset
    
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Collect V1 video shots in timeline order  
    shots = session.exec(
        select(Shot)
        .where(Shot.project_id == project_id)
        .where(Shot.track_index == 0)  # V1 only
        .order_by(Shot.start_frame.asc(), Shot.created_at.asc())
    ).all()
    
    # Collect video file paths
    video_files = []
    for shot in shots:
        if shot.video_url:
            from file_utils import resolve_path
            resolved = resolve_path(shot.video_url)
            if resolved and os.path.exists(resolved) and resolved.endswith(".mp4"):
                video_files.append(resolved)
    
    if not video_files:
        raise HTTPException(status_code=400, detail="No generated videos found for this project")
    
    # Collect A1 audio clips
    audio_files = []
    audio_shots = session.exec(
        select(Shot)
        .where(Shot.project_id == project_id)
        .where(Shot.track_index == 2)  # A1
        .order_by(Shot.start_frame.asc())
    ).all()
    for audio_shot in audio_shots:
        if audio_shot.video_url:
            from file_utils import resolve_path as rp
            resolved_audio = rp(audio_shot.video_url)
            if resolved_audio and os.path.exists(resolved_audio):
                audio_files.append(resolved_audio)
    
    render_id = f"render_{uuid.uuid4().hex[:8]}"
    
    # Run render in background
    background_tasks.add_task(
        _render_background, project_id, render_id, video_files, audio_files, project.fps
    )
    
    return {"job_id": render_id, "status": "rendering"}


async def _render_background(
    project_id: str, render_id: str,
    video_files: list, audio_files: list, fps: int
):
    """Background task: concat videos, mix audio, emit SSE progress."""
    import subprocess
    import asyncio
    from events import event_manager
    
    project_dir = os.path.join(config.PROJECTS_DIR, project_id, "generated")
    os.makedirs(project_dir, exist_ok=True)
    
    output_path = os.path.join(project_dir, f"{render_id}.mp4")
    temp_video_path = os.path.join(project_dir, f"{render_id}_video_only.mp4")
    list_file_path = os.path.join(project_dir, f"{render_id}_list.txt")
    
    total_steps = 2 + (1 if audio_files else 0)  # concat + finalize + optional audio mix
    current_step = 0
    
    async def emit_progress(step: int, message: str):
        pct = int((step / total_steps) * 100)
        await event_manager.broadcast("render_progress", {
            "job_id": render_id,
            "project_id": project_id,
            "progress": pct,
            "message": message
        })
    
    try:
        # Step 1: Concat videos
        await emit_progress(0, "Concatenating video clips...")
        
        with open(list_file_path, "w") as f:
            for mp4 in video_files:
                f.write(f"file '{mp4}'\n")
        
        concat_target = temp_video_path if audio_files else output_path
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_file_path,
            "-c", "copy", "-y", concat_target
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        current_step += 1
        await emit_progress(current_step, "Video concatenation complete")
        
        # Step 2: Mix audio (if any A1 clips exist) 
        if audio_files:
            await emit_progress(current_step, "Mixing audio track...")
            
            # Build ffmpeg command to overlay audio
            # For simplicity, take the first audio file and mix it in
            audio_input = audio_files[0]
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", temp_video_path,
                "-i", audio_input,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path
            ]
            subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            current_step += 1
            await emit_progress(current_step, "Audio mixed")
            
            # Clean up temp video-only file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        # Final: Clean up and emit completion
        if os.path.exists(list_file_path):
            os.remove(list_file_path)
        
        video_url = f"/projects/{project_id}/generated/{render_id}.mp4"
        
        await event_manager.broadcast("render_complete", {
            "job_id": render_id,
            "project_id": project_id,
            "video_url": video_url,
            "message": "Render complete!"
        })
        logger.info(f"Render complete: {output_path}")
        
    except Exception as e:
        logger.error(f"Render failed: {e}")
        await event_manager.broadcast("render_failed", {
            "job_id": render_id,
            "project_id": project_id,
            "error": str(e)
        })
        # Clean up partial files
        for f in [temp_video_path, list_file_path, output_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass


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
    
    # 3. Create New Shot (Copy of Original)
    new_shot_data = original_shot.model_dump(exclude={"id", "created_at", "updated_at"})
    new_shot = Shot(**new_shot_data)
    new_shot.id = uuid.uuid4().hex
    new_shot.created_at = datetime.utcnow()
    
    # 4. Configure New Shot (Trim In)
    # Starts at absolute_split_frame
    # trim_in_new = absolute_split_frame
    new_shot.trim_in = absolute_split_frame
    new_shot.trim_out = old_trim_out # Resets to original end
    
    # 5. Handle Track Positioning
    if original_shot.track_index == 0:
        # V1: Magnetic - Insert after original (Update timestamps is implicit in list order, but we track index?)
        # Current DB doesn't strictly use 'index' column for ordering, usually by created_at or implicit.
        # But we might need to shift things if we used 'start_frame'.
        # For V1, start_frame is ignored in rendering.
        pass
    else:
        # V2/A1: Free - Absolute Positioning
        # New shot starts where the cut happened
        # Start = Original Start + split_frame (seconds converted to frames?)
        # Wait, start_frame is absolute project frames.
        new_shot.start_frame = (original_shot.start_frame or 0) + (req.split_frame) # Shift start
    
    session.add(new_shot)
    
    # 6. Duplicate timeline/conditioning if necessary?
    # For now, simplistic copy. Ideally filter conditioning based on time?
    # TODO: Filter 'timeline' list based on split point (left vs right).
    
    session.commit()
    session.refresh(original_shot)
    session.refresh(new_shot)
    
    return {
        "original_shot": original_shot,
        "new_shot": new_shot
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
