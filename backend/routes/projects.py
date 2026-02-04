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
    
    # Sync Shots
    # Filter input fields to match Shot model columns
    valid_keys = Shot.model_fields.keys()
    
    # We replace shots? Or update?
    # Original logic: db_project.shots = new_shots (Replace)
    # But SQLModel relationship replacement can be tricky depending on cascade settings.
    # Safe approach: Delete old shots, add new ones (Primitive but safe for this scale)
    
    # Delete existing shots
    existing_shots = session.exec(select(Shot).where(Shot.project_id == project_id)).all()
    for s in existing_shots:
        session.delete(s)
        
    new_shots = []
    for s_data in state.shots:
        # s_data is a dict usually (from Pydantic dict())
        clean_data = {k: v for k, v in s_data.items() if k in valid_keys}
        clean_data["project_id"] = project_id # Force project_id from URL
        new_shots.append(Shot(**clean_data))
        
    session.add(db_project)
    for s in new_shots:
        session.add(s)
        
    session.commit()
        
    return {"status": "saved"}

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


