from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from schemas import (
    ScriptParseRequest, CommitStoryboardRequest,
    ReorderShotsRequest, AddShotRequest, BatchGenerateRequest,
    UpdateSceneRequest, BatchThumbnailRequest, ReorderScenesRequest
)
from typing import List
from sqlmodel import Session, select
from database import get_session, Scene, Shot, Project, Job
from storyboard.manager import StoryboardManager
from tasks.video import generate_video_task
import config
import os
import json
import uuid
import logging
from datetime import datetime, timezone

# Lazy import script_parser if needed, or import at top
from services.script_parser import script_parser
from services.ai_storyboard import ai_parse_script

logger = logging.getLogger(__name__)

router = APIRouter(tags=["storyboard"])


@router.post("/projects/{project_id}/script/parse")
async def parse_script(project_id: str, req: ScriptParseRequest):
    """Parses text into Scenes/Shots preview (no DB save)."""
    try:
        parsed_scenes = script_parser.parse_script(req.script_text, parse_mode=req.parse_mode or "auto")
        return {"scenes": parsed_scenes}
    except Exception as e:
        logger.error(f"Script parse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/storyboard/commit")
async def commit_storyboard(project_id: str, req: CommitStoryboardRequest, session: Session = Depends(get_session)):
    """Saves the parsed/edited storyboard structure to DB."""
    # Sync Storyboard (Smart Merge)
    # Goal: Preserve existing Scene/Shot IDs if they match structure (index/name).
    
    # 1. Load Existing Data
    existing_scenes = session.exec(select(Scene).where(Scene.project_id == project_id).order_by(Scene.index)).all()
    existing_shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    
    existing_scenes_map = {s.index: s for s in existing_scenes}
    
    # Pre-map existing shots: {scene_id: {shot_index: shot_obj}}
    existing_shots_map = {} 
    for s in existing_shots:
        if s.scene_id not in existing_shots_map: existing_shots_map[s.scene_id] = {}
        if s.index is not None:
             existing_shots_map[s.scene_id][s.index] = s
             
    # Track what we keep to delete the rest
    kept_scene_ids = set()
    kept_shot_ids = set()
    
    # 2. Process New Structure
    for i_scene, scene_data in enumerate(req.scenes):
        # Match Scene by Index
        if i_scene in existing_scenes_map:
            db_scene = existing_scenes_map[i_scene]
            # Update Content
            db_scene.name = scene_data.name or f"Scene {i_scene+1}"
            db_scene.script_content = scene_data.content
            session.add(db_scene)
        else:
            # Create New Scene
            db_scene = Scene(
                project_id=project_id,
                index=i_scene,
                name=scene_data.name or f"Scene {i_scene+1}",
                script_content=scene_data.content
            )
            session.add(db_scene)
            session.commit()  # Need ID for shots
            session.refresh(db_scene)
            
        kept_scene_ids.add(db_scene.id)
        
        # Match Shots within this Scene
        shots_data = scene_data.shots or []
        
        # Get existing shots for this scene
        scene_existing_shots = existing_shots_map.get(db_scene.id, {})
        
        for i_shot, shot_data in enumerate(shots_data):
            # Match Shot by Index
            if i_shot in scene_existing_shots:
                db_shot = scene_existing_shots[i_shot]
                # Update Content
                db_shot.action = shot_data.action
                db_shot.dialogue = shot_data.dialogue
                db_shot.character = shot_data.character
                db_shot.shot_type = shot_data.shot_type
                # Preserve Status, Video URL, etc.
                session.add(db_shot)
            else:
                # Create New Shot
                db_shot = Shot(
                    scene_id=db_scene.id,
                    project_id=project_id,
                    index=i_shot,
                    action=shot_data.action,
                    dialogue=shot_data.dialogue,
                    character=shot_data.character,
                    shot_type=shot_data.shot_type,
                    status="pending",
                    duration=4.0
                )
                session.add(db_shot)
            
            if not db_shot.id:
                 session.commit()
                 session.refresh(db_shot)

            kept_shot_ids.add(db_shot.id)

    # 3. Clean up orphans
    for s in existing_scenes:
        if s.id not in kept_scene_ids:
            session.delete(s)
            
    for s in existing_shots:
        if s.id not in kept_shot_ids:
            session.delete(s)
    
    session.commit()
    return {"status": "success", "message": f"Merged {len(req.scenes)} scenes."}

@router.get("/projects/{project_id}/storyboard")
async def get_storyboard(project_id: str, session: Session = Depends(get_session)):
    """Retrieve full hierarchy (Scenes -> Shots)."""
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id).order_by(Scene.index)).all()
    
    result = []
    for scene in scenes:
        shots = session.exec(select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.index)).all()
        
        scene_dict = scene.dict()
        scene_dict["shots"] = [s.dict() for s in shots]
        result.append(scene_dict)
        
    return {"scenes": result}


# ── Scene Management ────────────────────────────────────────────────


@router.patch("/projects/{project_id}/storyboard/scenes/{scene_id}")
async def update_scene(project_id: str, scene_id: str, req: UpdateSceneRequest, session: Session = Depends(get_session)):
    """Update a scene's name (Phase 1.5 fix: persist scene renames)."""
    scene = session.get(Scene, scene_id)
    if not scene or scene.project_id != project_id:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    if req.name is not None:
        scene.name = req.name
    session.add(scene)
    session.commit()
    return {"status": "success", "scene": scene.dict()}


@router.post("/projects/{project_id}/storyboard/scenes/reorder")
async def reorder_scenes(project_id: str, req: ReorderScenesRequest, session: Session = Depends(get_session)):
    """Reorder scenes within a project. scene_ids order becomes new Scene.index."""
    for new_index, scene_id in enumerate(req.scene_ids):
        scene = session.get(Scene, scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
        if scene.project_id != project_id:
            raise HTTPException(status_code=400, detail=f"Scene {scene_id} does not belong to project {project_id}")
        scene.index = new_index
        session.add(scene)
    
    session.commit()
    return {"status": "success", "message": f"Reordered {len(req.scene_ids)} scenes"}


# ── AI-Powered Endpoints ────────────────────────────────────────────


@router.post("/projects/{project_id}/storyboard/ai-parse")
async def ai_parse(project_id: str, req: ScriptParseRequest):
    """AI-powered script parsing using Gemma 3.
    
    Falls back to regex parser if Gemma is unavailable.
    """
    try:
        from model_engine import manager
        
        # Load pipeline (needed to access text encoder)
        pipeline = await manager.load_pipeline("ti2vid")
        text_encoder = pipeline.stage_1_model_ledger.text_encoder()
        
        parsed_scenes = ai_parse_script(
            text=req.script_text,
            text_encoder=text_encoder,
            seed=42,
        )
        
        # Cleanup
        del text_encoder
        from ltx_pipelines.utils.helpers import cleanup_memory
        cleanup_memory()
        
        return {"scenes": parsed_scenes, "mode": "ai"}
        
    except Exception as e:
        logger.warning(f"AI parse failed, falling back to regex: {e}")
        # Fallback to regex parser
        try:
            parsed_scenes = script_parser.parse_script(req.script_text, parse_mode="auto")
            return {"scenes": parsed_scenes, "mode": "fallback"}
        except Exception as e2:
            logger.error(f"Fallback parse also failed: {e2}")
            raise HTTPException(status_code=500, detail=str(e2))


@router.post("/projects/{project_id}/storyboard/generate-thumbnails")
async def generate_thumbnails(
    project_id: str,
    req: BatchThumbnailRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """Generate Flux 2 concept art thumbnails for shots without videos."""
    from tasks.image import generate_image_task
    
    results = []
    for shot_id in req.shot_ids:
        try:
            shot = session.get(Shot, shot_id)
            if not shot:
                results.append({"shot_id": shot_id, "status": "error", "detail": "Not found"})
                continue
            
            # Skip shots that already have a thumbnail or video
            if shot.thumbnail_url and not req.force:
                results.append({"shot_id": shot_id, "status": "skipped", "detail": "Already has thumbnail"})
                continue
            
            # Build prompt from shot data
            prompt = shot.action or shot.prompt or "A cinematic shot"
            if shot.character:
                prompt = f"{shot.character}: {prompt}"
            
            # Thumbnail generation params (small, fast)
            job_id = f"thumb_{uuid.uuid4().hex[:8]}"
            params = {
                "job_id": job_id,
                "project_id": project_id,
                "prompt": f"Cinematic storyboard concept art. {prompt}",
                "width": req.width or 512,
                "height": req.height or 320,
                "num_inference_steps": 15,  # Fast
                "cfg_scale": 2.0,
                "seed": shot.seed or 42,
                "shot_id": shot_id,  # So task can update shot thumbnail
                "is_thumbnail": True,
            }
            
            job = Job(
                id=job_id,
                project_id=project_id,
                type="thumbnail",
                status="pending",
                created_at=datetime.now(timezone.utc),
                prompt=params["prompt"],
                params_json=json.dumps(params),
            )
            session.add(job)
            
            # Mark shot as generating thumbnail
            shot.status = "generating_thumbnail"
            session.add(shot)
            session.commit()
            
            background_tasks.add_task(generate_image_task, job_id, params)
            results.append({"shot_id": shot_id, "status": "queued", "job_id": job_id})
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed for shot {shot_id}: {e}")
            results.append({"shot_id": shot_id, "status": "error", "detail": str(e)})
    
    return {"status": "success", "results": results}


# ── New Endpoints ────────────────────────────────────────────────────


@router.post("/projects/{project_id}/storyboard/shots/reorder")
async def reorder_shots(project_id: str, req: ReorderShotsRequest, session: Session = Depends(get_session)):
    """Reorder shots within a scene. shot_ids order becomes new index order."""
    # Validate scene exists
    scene = session.get(Scene, req.scene_id)
    if not scene or scene.project_id != project_id:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    for new_index, shot_id in enumerate(req.shot_ids):
        shot = session.get(Shot, shot_id)
        if not shot:
            raise HTTPException(status_code=404, detail=f"Shot {shot_id} not found")
        if shot.scene_id != req.scene_id:
            raise HTTPException(status_code=400, detail=f"Shot {shot_id} does not belong to scene {req.scene_id}")
        shot.index = new_index
        session.add(shot)
    
    session.commit()
    return {"status": "success", "message": f"Reordered {len(req.shot_ids)} shots"}


@router.post("/projects/{project_id}/storyboard/shots/add")
async def add_shot(project_id: str, req: AddShotRequest, session: Session = Depends(get_session)):
    """Add a manual shot to a scene."""
    scene = session.get(Scene, req.scene_id)
    if not scene or scene.project_id != project_id:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Find next index
    existing_shots = session.exec(
        select(Shot).where(Shot.scene_id == req.scene_id).order_by(Shot.index)
    ).all()
    next_index = (existing_shots[-1].index + 1) if existing_shots and existing_shots[-1].index is not None else 0
    
    # Get project defaults
    project = session.get(Project, project_id)
    
    new_shot = Shot(
        scene_id=req.scene_id,
        project_id=project_id,
        index=next_index,
        action=req.action,
        dialogue=req.dialogue,
        character=req.character,
        shot_type=req.shot_type,
        status="pending",
        duration=4.0,
        width=project.resolution_w if project else 768,
        height=project.resolution_h if project else 512,
        fps=project.fps if project else 25,
        seed=project.seed if project else 42,
    )
    session.add(new_shot)
    session.commit()
    session.refresh(new_shot)
    
    return {"status": "success", "shot": new_shot.dict()}


@router.delete("/projects/{project_id}/storyboard/shots/{shot_id}")
async def delete_storyboard_shot(project_id: str, shot_id: str, session: Session = Depends(get_session)):
    """Delete a shot and re-index remaining shots in that scene."""
    shot = session.get(Shot, shot_id)
    if not shot or shot.project_id != project_id:
        raise HTTPException(status_code=404, detail="Shot not found")
    
    scene_id = shot.scene_id
    session.delete(shot)
    session.commit()
    
    # Re-index remaining shots in the scene
    if scene_id:
        remaining = session.exec(
            select(Shot).where(Shot.scene_id == scene_id).order_by(Shot.index)
        ).all()
        for i, s in enumerate(remaining):
            s.index = i
            session.add(s)
        session.commit()
    
    return {"status": "success", "message": f"Shot {shot_id} deleted"}


@router.post("/projects/{project_id}/storyboard/batch-generate")
async def batch_generate(project_id: str, req: BatchGenerateRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """Queue generation for multiple shots sequentially."""
    results = []
    
    for shot_id in req.shot_ids:
        try:
            shot = session.get(Shot, shot_id)
            if not shot:
                results.append({"shot_id": shot_id, "status": "error", "detail": "Not found"})
                continue
            
            projects_dir = os.path.join(config.PROJECTS_DIR, project_id)
            manager = StoryboardManager(output_dir=projects_dir)
            
            job_config = await manager.prepare_shot_generation(shot_id, session)
            
            # Create Job
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            
            # Use shot's actual params instead of hardcoded values
            worker_params = {
                "job_id": job_id,
                "project_id": project_id,
                "prompt": job_config["prompt"],
                "width": shot.width,
                "height": shot.height,
                "num_frames": shot.num_frames,
                "fps": shot.fps,
                "num_inference_steps": 40,
                "cfg_scale": shot.cfg_scale,
                "pipeline_type": shot.pipeline_override or "advanced",
                "seed": shot.seed,
                "enhance_prompt": shot.enhance_prompt,
                "timeline": []
            }
            
            for img_path, idx, strength in job_config.get("images", []):
                worker_params["timeline"].append({
                    "path": img_path,
                    "frame_index": idx,
                    "strength": strength,
                    "type": "image"
                })
                
            worker_params["element_images"] = job_config.get("element_images", [])
            
            # Update Shot status
            shot.status = "generating"
            shot.prompt_enhanced = job_config["prompt"]
            session.add(shot)
            
            # Create Job Record
            job = Job(
                id=job_id,
                project_id=project_id,
                type="shot_generation",
                status="pending",
                created_at=datetime.now(timezone.utc),
                prompt=worker_params["prompt"],
                params_json=json.dumps(worker_params)
            )
            session.add(job)
            session.commit()
            
            background_tasks.add_task(generate_video_task, job_id, worker_params)
            results.append({"shot_id": shot_id, "status": "queued", "job_id": job_id})
            
        except Exception as e:
            logger.error(f"Batch generate failed for shot {shot_id}: {e}")
            results.append({"shot_id": shot_id, "status": "error", "detail": str(e)})
    
    return {"status": "success", "results": results}


@router.post("/shots/{shot_id}/generate")
async def generate_shot(shot_id: str, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """Trigger generation for a single shot."""
    try:
        shot = session.get(Shot, shot_id)
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")
        project_id = shot.project_id
            
        projects_dir = os.path.join(config.PROJECTS_DIR, project_id)
        manager = StoryboardManager(output_dir=projects_dir)
        
        job_config = await manager.prepare_shot_generation(shot_id, session)
        
        # Create Job
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Use shot's actual params, not hardcoded defaults
        worker_params = {
            "job_id": job_id,
            "project_id": project_id,
            "prompt": job_config["prompt"],
            "width": shot.width,
            "height": shot.height,
            "num_frames": shot.num_frames,
            "fps": shot.fps,
            "num_inference_steps": 40,
            "cfg_scale": shot.cfg_scale,
            "pipeline_type": shot.pipeline_override or "advanced",
            "seed": shot.seed,
            "enhance_prompt": shot.enhance_prompt,
            "timeline": [] 
        }
        
        for img_path, idx, strength in job_config.get("images", []):
            worker_params["timeline"].append({
                "path": img_path,
                "frame_index": idx,
                "strength": strength,
                "type": "image"
            })
            
        # Add Element Visuals (for IP-Adapter)
        worker_params["element_images"] = job_config.get("element_images", [])
        
        # Update Shot status
        shot.status = "generating"
        shot.prompt_enhanced = job_config["prompt"] 
        session.add(shot)
        
        # Create Job Record
        job = Job(
            id=job_id,
            project_id=project_id,
            type="shot_generation",
            status="pending",
            created_at=datetime.now(timezone.utc),
            prompt=worker_params["prompt"],
            params_json=json.dumps(worker_params) 
        )
        session.add(job)
        session.commit()
        
        # Trigger Worker
        background_tasks.add_task(generate_video_task, job_id, worker_params)
        
        return {"status": "queued", "job_id": job_id}
            
    except Exception as e:
        logger.error(f"Shot generation trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
