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
    # Clear existing structure? Or Append? 
    # For MVP: Clear existing scenes/shots for project to avoid dupes
    existing_scenes = session.exec(select(Scene).where(Scene.project_id == project_id)).all()
    for s in existing_scenes:
        session.delete(s)
    existing_shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    for s in existing_shots:
        session.delete(s)
    
    # Save new structure
    # req.scenes is list of dicts from ParsedScene
    for i_scene, scene_data in enumerate(req.scenes):
        db_scene = Scene(
            project_id=project_id,
            index=i_scene,
            name=scene_data.get("name", f"Scene {i_scene+1}"),
            script_content=scene_data.get("content")
        )
        session.add(db_scene)
        session.commit()
        session.refresh(db_scene) # Get ID
        
        # Save Shots
        shots = scene_data.get("shots", [])
        for i_shot, shot_data in enumerate(shots):
            # shot_data matches ParsedShot (dict)
            db_shot = Shot(
                scene_id=db_scene.id,
                project_id=project_id,
                index=i_shot,
                action=shot_data.get("action"),
                dialogue=shot_data.get("dialogue"),
                character=shot_data.get("character"),
                # Defaults
                status="pending",
                duration=4.0
            )
            session.add(db_shot)
    
    session.commit()
    return {"status": "success", "message": f"Committed {len(req.scenes)} scenes."}

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

