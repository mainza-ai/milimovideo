from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from schemas import ScriptParseRequest, CommitStoryboardRequest
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["storyboard"])



@router.post("/projects/{project_id}/script/parse")
async def parse_script(project_id: str, req: ScriptParseRequest):
    """Parses text into Scenes/Shots preview (no DB save)."""
    try:
        parsed_scenes = script_parser.parse_script(req.script_text)
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
    # Map Shots by (SceneID, Index) is tricky if SceneID changes.
    # Map Shots by ID? No, the frontend "Script" doesn't know IDs yet (it parses text).
    # It sends "Action", "Dialogue".
    # We must match by INDEX within Scene Index.
    # Structure: Scene[i] -> Shot[j].
    
    # Pre-map existing shots: {scene_id: {shot_index: shot_obj}}
    existing_shots_map = {} 
    for s in existing_shots:
        if s.scene_id not in existing_shots_map: existing_shots_map[s.scene_id] = {}
        # Use existing index. If index is None, we can't map reliably.
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
            db_scene.name = scene_data.get("name", f"Scene {i_scene+1}")
            db_scene.script_content = scene_data.get("content")
            session.add(db_scene)
        else:
            # Create New Scene
            db_scene = Scene(
                project_id=project_id,
                index=i_scene,
                name=scene_data.get("name", f"Scene {i_scene+1}"),
                script_content=scene_data.get("content")
            )
            session.add(db_scene)
            session.commit() # Need ID for shots
            session.refresh(db_scene)
            
        kept_scene_ids.add(db_scene.id)
        
        # Match Shots within this Scene
        shots_data = scene_data.get("shots", [])
        
        # Get existing shots for this scene (if it existed)
        # Verify if scene changed ID (it shouldn't if we reused obj)
        scene_existing_shots = existing_shots_map.get(db_scene.id, {})
        
        for i_shot, shot_data in enumerate(shots_data):
            # Match Shot by Index
            if i_shot in scene_existing_shots:
                db_shot = scene_existing_shots[i_shot]
                # Update Content
                db_shot.action = shot_data.get("action")
                db_shot.dialogue = shot_data.get("dialogue")
                db_shot.character = shot_data.get("character")
                # Preserve Status, Video URL, etc.
                session.add(db_shot)
            else:
                # Create New Shot
                db_shot = Shot(
                    scene_id=db_scene.id,
                    project_id=project_id,
                    index=i_shot,
                    action=shot_data.get("action"),
                    dialogue=shot_data.get("dialogue"),
                    character=shot_data.get("character"),
                    status="pending",
                    duration=4.0
                )
                session.add(db_shot)
            
            # Flush to get ID if needed? No, commit at end is fine unless we need FK.
            # SQLModel handles relationship if object attached?
            # We are setting scene_id explicitly.
            if not db_shot.id:
                 session.commit()
                 session.refresh(db_shot)

            kept_shot_ids.add(db_shot.id)

    # 3. Clean up orphans
    # Scenes
    for s in existing_scenes:
        if s.id not in kept_scene_ids:
            session.delete(s)
            
    # Shots (Global cleanup for this project)
    # Re-fetch or iterate initial list?
    # Initial list might contain shots from deleted scenes.
    for s in existing_shots:
        if s.id not in kept_shot_ids:
            session.delete(s)
    
    session.commit()
    return {"status": "success", "message": f"Merged {len(req.scenes)} scenes."}

@router.get("/projects/{project_id}/storyboard")
async def get_storyboard(project_id: str, session: Session = Depends(get_session)):
    """Retrieve full hierarchy (Scenes -> Shots)."""
    # Get Scenes
    scenes = session.exec(select(Scene).where(Scene.project_id == project_id).order_by(Scene.index)).all()
    
    result = []
    for scene in scenes:
        # Get Shots for scene
        shots = session.exec(select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.index)).all()
        
        scene_dict = scene.dict()
        scene_dict["shots"] = [s.dict() for s in shots]
        result.append(scene_dict)
        
    return {"scenes": result}

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
        
        # Map Shot Config to Worker Params
        worker_params = {
            "job_id": job_id,
            "project_id": project_id,
            "prompt": job_config["prompt"],
            "width": 768, "height": 512, "num_frames": 121, "fps": 25,
            "num_inference_steps": 40,
            "pipeline_type": "advanced",
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

