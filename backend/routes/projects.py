from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
from sqlmodel import Session, select
from database import get_session, Project, Shot
from schemas import CreateProjectRequest, ProjectState
import uuid
import os
import shutil
import logging
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
        seed=new_project.seed
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
            seed=p.seed
        ))
    return results

@router.get("/projects/{project_id}", response_model=ProjectState)
def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Load shots
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
            "auto_continue": False,
            "status": s.status,
            "image_url": s.thumbnail_url,
            "video_url": s.video_url,
            "created_at": s.created_at.isoformat() if s.created_at else None
        })
        
    return ProjectState(
        id=project.id,
        name=project.name,
        shots=shots_data,
        resolution_w=project.resolution_w,
        resolution_h=project.resolution_h,
        fps=project.fps,
        seed=project.seed
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
            raise HTTPException(status_code=404, detail="Project not found")
    
    db_project.name = state.name
    
    # Save settings
    db_project.resolution_w = state.resolution_w
    db_project.resolution_h = state.resolution_h
    db_project.fps = state.fps
    db_project.seed = state.seed
    
    # db_project.updated_at = datetime.now(timezone.utc) # Handled by SQLModel typically or trigger
    
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
                # Don't overwrite generated fields if they are missing/null in payload
                # but payload usually sends what it has.
                # Crucial: Frontend might send "video_url": null if it doesn't know about it (unlikely if it loaded project first)
                # But to be safe, we only overwrite if value is provided OR if we explicitly want to clear it.
                # However, usually we trust the payload. 
                # The RISK is if frontend *recreates* the shot object and loses the ID or properties.
                # If ID matches, we assume frontend knows the state.
                # BUT: `video_url` and `status` should be preserved if the payload has them as null/undefined?
                # Let's assume frontend sends full object state.
                # The issue described in the audit was: Storyboard Re-Commit *Deletes* shots because it *regenerates* the ID or just wipes DB.
                # Here in `save_project`, the frontend sends the IDs it knows.
                
                # PROTECT generated fields if payload is empty for them?
                # Actually, if we just `setattr` everything, we trust frontend.
                # Providing Smart Merge for `save_project` means we trust the ID linkage.
                setattr(existing_shot, k, v)
                
            # Restore protected fields if they were wiped by frontend sending partial data (safety net)
            # (Optional, depends on how "ProjectState" is constructed in frontend)
            # For now, standard update is fine as long as ID persists.
            session.add(existing_shot)
        else:
            # CREATE New Shot
            # Ensure ID is set
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
        
    return {"status": "saved"}

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
    Stitches all generated shots in the project into a final MP4.
    """
    from database import Job
    import subprocess
    
    # 1. Load project from DB
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    shots = session.exec(select(Shot).where(Shot.project_id == project_id).order_by(Shot.created_at)).all()
    if not shots:
        raise HTTPException(status_code=400, detail="Project has no shots")
    
    # 2. Collect video paths by looking up jobs in DB
    input_files = []
    
    for shot in shots:
        if not shot.last_job_id:
            continue
        
        # Look up job to get actual output path
        job = session.get(Job, shot.last_job_id)
        if job and job.output_path:
            # Require project-scoped paths
            if not job.output_path.startswith("/projects/"):
                logger.warning(f"Skipping job {shot.last_job_id} in render - uses legacy path format")
                continue
            
            # config.PROJECTS_DIR is base for project paths
            # path is like /projects/{id}/generated/foo.mp4
            rel = job.output_path.lstrip("/projects/")
            file_path = os.path.join(config.PROJECTS_DIR, rel)
            
            if os.path.exists(file_path) and file_path.endswith(".mp4"):
                input_files.append(file_path)
    
    if not input_files:
        raise HTTPException(status_code=400, detail="No generated videos found for this project")
    
    # 3. Stitch to project workspace
    render_job_id = f"render_{uuid.uuid4().hex[:8]}"
    
    # Output to project's generated folder
    project_generated_dir = os.path.join(config.PROJECTS_DIR, project_id, "generated")
    os.makedirs(project_generated_dir, exist_ok=True)
    output_path = os.path.join(project_generated_dir, f"{render_job_id}.mp4")
    
    try:
        list_file_path = os.path.join(project_generated_dir, f"{render_job_id}_list.txt")
        with open(list_file_path, "w") as f:
            for mp4 in input_files:
                f.write(f"file '{mp4}'\n")
        
        # ffmpeg concat
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path,
            "-c", "copy", "-y", output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Cleanup list file
        if os.path.exists(list_file_path):
            os.remove(list_file_path)
        
        # Return project-scoped URL
        return {"status": "completed", "video_url": f"/projects/{project_id}/generated/{render_job_id}.mp4"}
    
    except Exception as e:
        print(f"Render failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/images")
async def get_project_images(project_id: str, session: Session = Depends(get_session)):
    """List all generated images for a project."""
    from database import Asset
    statement = select(Asset).where(Asset.project_id == project_id, Asset.type == "image").order_by(Asset.created_at.desc())
    results = session.exec(statement).all()
    return results


