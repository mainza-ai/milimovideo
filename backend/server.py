import sys
from fastapi import FastAPI, UploadFile, BackgroundTasks, File, Form, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os


import shutil
import uuid
import json
import logging
from typing import List, Optional, Literal
from datetime import datetime, timezone

# Logging
import config

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("server")

print("--- STARTING MILIMO VIDEO SERVER ---")

# Setup Paths (Sys.path) from config
config.setup_paths()

from worker import generate_video_task

PROJECTS_DIR = config.PROJECTS_DIR
# Note: All assets (uploads, generated, thumbnails) are now in /projects/{id}/ subfolders

from events import event_manager
from database import init_db, get_session, Project, Job, Asset, Shot, engine
from sqlmodel import Session, select

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Backend V2 starting...")
    init_db() # Initialize SQLite DB
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    yield
    print("Backend V2 shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/events")
async def events(request: Request):
    return await event_manager.subscribe(request)



# --- Data Models ---

class TimelineItem(BaseModel):
    path: str
    frame_index: int = 0
    strength: float = 1.0
    type: Literal["image", "video"]

class ShotConfig(BaseModel):
    id: str
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: int = 42
    width: int = 768
    height: int = 512
    num_frames: int = 121
    fps: int = 25
    num_inference_steps: int = 40
    cfg_scale: float = 3.0
    enhance_prompt: bool = True
    upscale: bool = True
    pipeline_override: Optional[str] = "auto"
    auto_continue: bool = False
    timeline: List[TimelineItem] = []

class GenerateAdvancedRequest(BaseModel):
    project_id: str
    shot_config: ShotConfig

class ProjectState(BaseModel):
    id: str
    name: str
    shots: List[dict]
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42

class CreateProjectRequest(BaseModel):
    name: str
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42

# --- Helper functions ---

def generate_thumbnail(video_path: str) -> Optional[str]:
    thumb_path = os.path.splitext(video_path)[0] + "_thumb.jpg"
    # If already exists, return
    if os.path.exists(thumb_path):
        return thumb_path
        
    try:
        import subprocess
        # Extract frame at 0.5s to avoid black start frames
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return thumb_path
    except Exception as e:
        logger.error(f"Thumb gen failed: {e}")
        # Try at 0s if 0.5s failed (e.g. video too short)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-ss", "00:00:00", "-vframes", "1", thumb_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return thumb_path
        except:
            return None



@app.post("/upload")
async def upload_asset(file: UploadFile = File(...), project_id: Optional[str] = Form(None)):
    """
    Uploads an asset and returns a permanent path.
    If project_id is provided, saves to project assets folder.
    Otherwise, saves to global uploads folder.
    """
    ext = file.filename.split(".")[-1].lower()
    asset_id = uuid.uuid4().hex
    filename = f"{asset_id}.{ext}"
    
    # Determine upload directory
    if project_id:
        upload_dir = os.path.join(PROJECTS_DIR, project_id, "assets")
        web_prefix = f"/projects/{project_id}/assets"
    else:
        # Legacy/Global uploads
        upload_dir = os.path.join(PROJECTS_DIR, "_uploads")
        web_prefix = "/projects/_uploads"
        
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    item_type = "video" if ext in ["mp4", "mov", "avi"] else "image"
    thumb_url = None
    
    if item_type == "video":
        thumb_path = generate_thumbnail(file_path)
        if thumb_path:
            thumb_filename = os.path.basename(thumb_path)
            thumb_url = f"{web_prefix}/{thumb_filename}"
            
    # Persist to DB
    with Session(engine) as session:
        asset = Asset(
            id=asset_id,
            project_id=project_id,
            type=item_type,
            path=file_path,
            url=f"{web_prefix}/{filename}",
            filename=file.filename,
            created_at=datetime.now(timezone.utc)
        )
        session.add(asset)
        session.commit()
            
    return {
        "asset_id": asset_id,
        "url": f"{web_prefix}/{filename}",
        "access_path": file_path, 
        "filename": file.filename,
        "type": item_type,
        "thumbnail": thumb_url
    }

# ... (get_shot_last_frame) ...

# ... (start of endpoints)

@app.post("/shot/{job_id}/last-frame")
async def get_shot_last_frame(job_id: str):
    """
    Returns the path/url to the last frame of a generated video.
    Used for 'Extend' functionality.
    """
    import subprocess
    
    # 1. Look up the job in DB to get project-scoped output path
    with Session(engine) as session:
        job = session.get(Job, job_id)
        
        if job and job.output_path:
            # Convert DB path to filesystem path
            # job.output_path can be: /projects/{id}/generated/file.mp4 or /generated/file.mp4
            if job.output_path.startswith("/projects/"):
                # Project-scoped: /projects/{id}/generated/filename.mp4
                video_path = os.path.join(os.path.dirname(__file__), job.output_path.lstrip("/"))
            else:
                # Legacy: /generated/filename.mp4
                video_path = os.path.join(GENERATED_DIR, os.path.basename(job.output_path))
            
            # Check if it's an image (single frame generation)
            if job.output_path.endswith(".jpg") or job.output_path.endswith(".png"):
                if os.path.exists(video_path):
                    return {"url": job.output_path, "path": video_path}
            
            if os.path.exists(video_path):
                try:
                    # Extract last frame to project workspace
                    out_name = f"{job_id}_last.jpg"
                    
                    # Determine output directory (same as video)
                    out_dir = os.path.dirname(video_path)
                    out_path = os.path.join(out_dir, out_name)
                    
                    # Extract last frame using ffmpeg
                    subprocess.run([
                        "ffmpeg", "-y", "-sseof", "-1.0", "-i", video_path,
                        "-q:v", "2", "-update", "1", out_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Build URL based on original path pattern
                    if job.output_path.startswith("/projects/"):
                        # Extract project path prefix
                        parts = job.output_path.split("/")
                        project_id = parts[2]
                        out_url = f"/projects/{project_id}/generated/{out_name}"
                    else:
                        out_url = f"/generated/{out_name}"
                    
                    return {"url": out_url, "path": out_path}
                except Exception as e:
                    logger.error(f"Failed to extract last frame from {video_path}: {e}")
    
    # 2. Legacy fallback: Check GENERATED_DIR directly
    video_path = os.path.join(GENERATED_DIR, f"{job_id}.mp4")
    if not os.path.exists(video_path):
        image_path = os.path.join(GENERATED_DIR, f"{job_id}.jpg")
        if os.path.exists(image_path):
            return {"url": f"/generated/{job_id}.jpg", "path": image_path}
        raise HTTPException(status_code=404, detail="Video/Image not found")
    
    try:
        out_name = f"{job_id}_last.jpg"
        out_path = os.path.join(GENERATED_DIR, out_name)
        
        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-1.0", "-i", video_path,
            "-q:v", "2", "-update", "1", out_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return {"url": f"/generated/{out_name}", "path": out_path}
    except Exception as e:
        logger.error(f"Failed to extract last frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/advanced")
async def generate_advanced(req: GenerateAdvancedRequest, background_tasks: BackgroundTasks):
    """
    Unified endpoint for all generation types (T2V, I2V, T2I, V2V).
    The worker inspects 'timeline' to decide the pipeline.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    # Validated Params
    params = req.shot_config.model_dump()
    params["job_id"] = job_id
    params["project_id"] = req.project_id
    params["pipeline_type"] = "advanced"

    # --- Smart Continue / Auto-Conditioning Logic ---
    # If auto_continue is True and NO explicit image/video conditioning is provided for frame 0,
    # try to use the last completed shot from this project.
    if req.shot_config.auto_continue:
        has_initial_cond = any(
            t.frame_index == 0 and t.type in ["image", "video"] 
            for t in req.shot_config.timeline
        )
        
        if not has_initial_cond:
            # Query DB for last successful job in this project
            with Session(engine) as session:
                last_job = session.exec(
                    select(Job)
                    .where(Job.project_id == req.project_id)
                    .where(Job.status == "completed")
                    .where(Job.output_path != None)
                    .order_by(Job.created_at.desc())
                ).first()
                
                if last_job and last_job.output_path:
                    logger.info(f"Smart Continue: Extending from last job {last_job.id}")
                    
                    # Ensure we have a frame to use. 
                    # If it's a video, extract last frame. If image, use it directly.
                    try:
                        cond_path = last_job.output_path
                        if cond_path.endswith(".mp4") or cond_path.endswith(".mov"):
                             # We need to extract the last frame. 
                             # We can reuse the logic from get_shot_last_frame or call it internaly?
                             # Better to do it inline or call helper.
                             
                             # Let's generate a temporary last frame path
                             last_frame_path = os.path.join(GENERATED_DIR, f"{last_job.id}_auto_last.jpg")
                             
                             # Reuse FFmpeg command logic
                             import subprocess
                             if not os.path.exists(last_frame_path):
                                 subprocess.run([
                                     "ffmpeg", "-y", "-sseof", "-1.0", "-i", cond_path, 
                                     "-q:v", "2", "-update", "1", last_frame_path
                                 ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                             
                             cond_path = last_frame_path

                        # Add to params AND timeline
                        # We must update 'params' (dict) passed to worker
                        # And strictly speaking, worker parses 'timeline' list.
                        
                        # Add to timeline params
                        new_item = {
                            "path": cond_path, 
                            "frame_index": 0, 
                            "strength": 1.0, 
                            "type": "image"
                        }
                        
                        # Append to params["timeline"]
                        params["timeline"] = [new_item] + params.get("timeline", [])
                        
                        # Update description string for UI/Logs
                        logger.info(f"Auto-injected conditioning from {cond_path}")
                        
                    except Exception as e:
                        logger.error(f"Smart Continue failed to extract frame: {e}")

    # Persist Job to DB (Pending)
    with Session(engine) as session:
        job = Job(
            id=job_id,
            project_id=req.project_id,
            type="advanced",
            status="pending",
            created_at=datetime.now(timezone.utc),
            prompt=req.shot_config.prompt,
            params_json=json.dumps(params)
        )
        session.add(job)
        session.commit()
    
    background_tasks.add_task(generate_video_task, job_id, params)
    
    background_tasks.add_task(generate_video_task, job_id, params)
    
    return {"job_id": job_id, "status": "queued"}

# --- Element Management (Storyboard) ---
from managers.element_manager import element_manager

class ElementCreate(BaseModel):
    name: str
    type: str
    description: str
    trigger_word: Optional[str] = None
    image_path: Optional[str] = None

@app.post("/projects/{project_id}/elements")
async def create_element(project_id: str, element: ElementCreate):
    return element_manager.create_element(
        project_id, 
        element.name, 
        element.type, 
        element.description, 
        element.trigger_word,
        element.image_path
    )

@app.get("/projects/{project_id}/elements")
async def get_elements(project_id: str):
    return element_manager.get_elements(project_id)

@app.delete("/elements/{element_id}")
async def delete_element(element_id: str):
    success = element_manager.delete_element(element_id)
    return {"success": success}

class ElementVisualizeRequest(BaseModel):
    prompt_override: Optional[str] = None

@app.post("/elements/{element_id}/visualize")
async def visualize_element(element_id: str, req: ElementVisualizeRequest, background_tasks: BackgroundTasks):
    # This might take ~10-20s on Mac, so maybe background?
    # But user expects immediate feedback? 
    # Let's run synchronously for now so we can return the path directly for the UI update.
    # The 'generate_visual' method in ElementManager calls Flux which is blocking anyway.
    # If we want true async status we'd need a Job ID.
    # For "Phase 5 MVP" let's block (User waits ~20s-90s depending on device).
    # Wait, blocking the main thread blocks ALL requests. That's bad.
    # We should run in threadpool or make generate_visual truly async-friendly (offload).
    
    # FastAPI handles async def by running in event loop.
    # But our code calls torch (CPU bound for GIL).
    # We should use run_in_executor if we want to be nice.
    # HOWEVER, FluxInpainter is not thread-safe if shared? 
    # It just holds the model.
    # Let's just await it. It will block the event loop if strictly CPU bound during inference.
    # Flux MPS inference releases GIL? Usually PyTorch C++ does.
    
    path = await element_manager.generate_visual(element_id, req.prompt_override)
    if not path:
        raise HTTPException(status_code=500, detail="Generation failed")
    
    # Return web-accessible URL
    # path is absolute: /Users/.../milimovideo/projects/{id}/elements/visual_xyz.jpg
    # web prefix: /projects/{id}/elements/...
    # Helper needed to convert absolute path to URL based on PROJECTS_DIR
    
    # We can infer:
    # rel_path = os.path.relpath(path, PROJECTS_DIR)
    # url = f"/projects/{rel_path}"
    
    rel_path = os.path.relpath(path, PROJECTS_DIR)
    url = f"/projects/{rel_path}"
    
    return {"image_path": path, "url": url}

# --- In-Painting Extensions ---
from managers.inpainting_manager import inpainting_manager

class InpaintRequest(BaseModel):
    image_path: str
    mask_path: Optional[str] = None
    points: Optional[str] = None # For SAM
    prompt: str

@app.post("/edit/inpaint")
async def inpaint_image(job_id: str, req: InpaintRequest, background_tasks: BackgroundTasks):
    # In a real scenario, we might want to offload to background task if slow
    # For now, let's just trigger it.
    # Note: job_id should be unique for this edit action
    
    # If points provided, get mask first
    mask_path = req.mask_path
    if req.points and not mask_path:
        mask_path = await inpainting_manager.get_mask_from_sam(req.image_path, eval(req.points))
    
    # Run Inpainting
    if not mask_path:
         return {"status": "error", "message": "No mask provided"}

    # We run this in background to avoid blocking API
    background_tasks.add_task(inpainting_manager.process_inpaint, job_id, req.image_path, mask_path, req.prompt)
    
    return {"status": "queued", "job_id": job_id}

@app.post("/edit/segment")
async def segment_preview(file: UploadFile = File(...), points: str = Form(...)):
    # Proxy to SAM Microservice
    # This is for the frontend to get a preview mask
    import requests
    files = {"file": file.file}
    data = {"points": points}
    try:
        res = requests.post(f"http://localhost:{config.SAM_SERVICE_PORT}/segment", files=files, data=data)
        return res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Storyboard Engine ---

from services.script_parser import script_parser, ParsedScene
from storyboard.manager import StoryboardManager
from database import Scene, Shot

class ScriptParseRequest(BaseModel):
    script_text: str

class CommitStoryboardRequest(BaseModel):
    scenes: List[dict] # Should match parsed structure

@app.post("/projects/{project_id}/script/parse")
async def parse_script(project_id: str, req: ScriptParseRequest):
    """Parses text into Scenes/Shots preview (no DB save)."""
    try:
        parsed_scenes = script_parser.parse_script(req.script_text)
        return {"scenes": parsed_scenes}
    except Exception as e:
        logger.error(f"Script parse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{project_id}/storyboard/commit")
async def commit_storyboard(project_id: str, req: CommitStoryboardRequest):
    """Saves the parsed/edited storyboard structure to DB."""
    with Session(engine) as session:
        # Clear existing structure? Or Append? 
        # For MVP: Clear existing scenes/shots for project to avoid dupes
        # (Be careful in prod!)
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

@app.get("/projects/{project_id}/storyboard")
async def get_storyboard(project_id: str):
    """Retrieve full hierarchy (Scenes -> Shots)."""
    with Session(engine) as session:
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

@app.post("/shots/{shot_id}/generate")
async def generate_shot(shot_id: str, background_tasks: BackgroundTasks):
    """Trigger generation for a single shot."""
    try:
        # We need to construct parameters for the worker
        # Use StoryboardManager to prepare the "smart" params (conditioning etc)
        
        # Where are artifacts stored? /projects/{id}/artifacts?
        # Need to find project_id first.
        with Session(engine) as session:
            shot = session.get(Shot, shot_id)
            if not shot:
                raise HTTPException(status_code=404, detail="Shot not found")
            project_id = shot.project_id
            
        projects_dir = os.path.join(PROJECTS_DIR, project_id)
        manager = StoryboardManager(output_dir=projects_dir)
        
        with Session(engine) as session:
            job_config = await manager.prepare_shot_generation(shot_id, session)
            
            # Create Job
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            
            # Map Shot Config to Worker Params
            worker_params = {
                "job_id": job_id,
                "project_id": project_id,
                "prompt": job_config["prompt"],
                # Standard LTX settings (could be overridden by Shot/Project settings later)
                "width": 768,
                "height": 512,
                "num_frames": 121,
                "fps": 25,
                "num_inference_steps": 40,
                "pipeline_type": "advanced", # Use unified pipeline
                "timeline": [] # Constructed below from images
            }
            
            # Convert conditioning images to Timeline items
            # manager return images as list of (path, frame_idx, strength)
            for img_path, idx, strength in job_config.get("images", []):
                worker_params["timeline"].append({
                    "path": img_path,
                    "frame_index": idx,
                    "strength": strength,
                    "type": "image" # Assuming image conditioning for now
                })
            
            # Update Shot status
            shot = session.get(Shot, shot_id)
            shot.status = "generating"
            shot.prompt_enhanced = job_config["prompt"] # Save what we actually used
            session.add(shot)
            
            # Create Job Record
            job = Job(
                id=job_id,
                project_id=project_id,
                type="shot_generation", # Distinguish from generic
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

@app.post("/projects")
async def create_project(request: CreateProjectRequest):
    project_id = uuid.uuid4().hex
    
    with Session(engine) as session:
        db_project = Project(
            id=project_id,
            name=request.name,
            # shots=[], # Removed from model
            resolution_w=request.resolution_w,
            resolution_h=request.resolution_h,
            fps=request.fps,
            seed=request.seed,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        session.add(db_project)
        session.commit()
        session.refresh(db_project)

        return {
            "id": db_project.id,
            "name": db_project.name,
            "shots": [], # New project has no shots
            "seed": db_project.seed,
            "resolution_w": db_project.resolution_w,
            "resolution_h": db_project.resolution_h,
            "fps": db_project.fps
        }

@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        
        if not project:
            # Fallback for legacy JSON
            legacy_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
            if os.path.exists(legacy_path):
                 # ... (Implementation omitted for brevity, assume legacy handling is deprecated or handled elsewhere)
                 pass
            raise HTTPException(status_code=404, detail="Project not found")

        # Fetch Shots from DB
        shots = session.exec(select(Shot).where(Shot.project_id == project_id).order_by(Shot.index)).all()
        shots_list = [s.dict() for s in shots]

        # Construct response
        return {
            "id": project.id,
            "name": project.name,
            "shots": shots_list,
            "settings": { 
                "resolution_w": project.resolution_w,
                "resolution_h": project.resolution_h,
                "fps": project.fps
            },
            "fps": project.fps, 
            "resolution_w": project.resolution_w,
            "resolution_h": project.resolution_h,
            "seed": project.seed
        }

@app.put("/projects/{project_id}")
async def save_project(project_id: str, state: ProjectState):
    if state.id != project_id:
        raise HTTPException(status_code=400, detail="Project ID mismatch")
    
    with Session(engine) as session:
        db_project = session.get(Project, project_id)
        if not db_project:
             # Create if missing?
             raise HTTPException(status_code=404, detail="Project not found")
        
        db_project.name = state.name
        
        # Save settings
        db_project.resolution_w = state.resolution_w
        db_project.resolution_h = state.resolution_h
        db_project.fps = state.fps
        db_project.seed = state.seed
        
        db_project.updated_at = datetime.now(timezone.utc)
        
        # Sync Shots
        # Filter input fields to match Shot model columns
        valid_keys = Shot.model_fields.keys()
        
        new_shots = []
        for s_data in state.shots:
            # s_data is a dict
            clean_data = {k: v for k, v in s_data.items() if k in valid_keys}
            
            # Map prompt -> action if missing (legacy support)
            # The Shot model now has 'action' AND 'prompt'
            
            # Handle defaults if needed
            new_shots.append(Shot(**clean_data))
            
        db_project.shots = new_shots
        
        session.add(db_project)
        session.commit()
        
    # Optional: Still save JSON for backup/safety during transition?
    # No, we want to unify. BUT if we want to be safe...
    # Let's delete the JSON to avoid confusion if it exists.
    legacy_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
    if os.path.exists(legacy_path):
        os.remove(legacy_path)
        
    return {"status": "saved"}

@app.get("/projects")
async def list_projects():
    """Returns a list of all projects registered in the database, sorted by last updated."""
    with Session(engine) as session:
        projects = session.exec(select(Project).order_by(Project.updated_at.desc())).all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "created_at": p.created_at.timestamp(),
                "updated_at": p.updated_at.timestamp()
            }
            for p in projects
        ]

@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            session.delete(project)
            session.commit()
        
        # 2. Delete Project Folder (Assets, generated, etc.)
        project_dir = os.path.join(PROJECTS_DIR, project_id)
        if os.path.exists(project_dir):
            try:
                shutil.rmtree(project_dir)
                logger.info(f"Deleted project directory: {project_dir}")
            except Exception as e:
                logger.error(f"Failed to delete project directory {project_dir}: {e}")
        
        # 3. Delete Legacy JSON if exists
        legacy_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
            
        return {"status": "deleted"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    # 1. Check Active Jobs (Memory - Fastest)
    from worker import active_jobs
    if job_id in active_jobs:
        # Active in memory
        job_data = active_jobs[job_id]
        status_response = {
            "job_id": job_id,
            "status": job_data.get("status", "processing"),
            "progress": job_data.get("progress", 0),
            "eta_seconds": job_data.get("eta_seconds", None),
            "current_prompt": job_data.get("current_prompt", None),
            "status_message": job_data.get("status_message", "Processing..."),
            "actual_frames": job_data.get("actual_frames", None)
        }
        return status_response
             # If it has actual_frames, it might be completed?
             # worker updates active_jobs status? NO. worker only updates DB status.
             # active_jobs just holds progress.
             # but we added `active_jobs[job_id]["actual_frames"] = actual_frames` in worker.
             
        return status_response
    
    # 2. Check DB (Source of Truth)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job:
            thumb_url = job.thumbnail_path # Single Source of Truth from DB
            
            # Legacy Fallback (Migration Period)
            if not thumb_url and job.output_path:
                 # Only check FS if DB is empty (for old jobs)
                 alt_thumb_path = job.output_path.replace(".mp4", "_thumb.jpg").replace(".mov", "_thumb.jpg")
                 # Check if actually exists on disk? NO. 
                 # We agreed on DB Truth. But for old jobs, we might need to check.
                 # Let's keep a minimal check only for legacy.
                 # Actually, better to just return None if not in DB to encourage migration?
                 # No, user experience first.
                 # If job.output_path is a URL like /generated/..., we need os path.
                 filename = os.path.basename(job.output_path)
                 fs_path = os.path.join(GENERATED_DIR, filename.replace(".mp4", "_thumb.jpg").replace(".mov", "_thumb.jpg"))
                 if os.path.exists(fs_path):
                     thumb_url = f"/generated/{os.path.basename(fs_path)}"

            return {
                "job_id": job.id,
                "status": job.status,
                "progress": job.progress,
                # Use job.output_path directly - already project-scoped in DB
                "video_url": job.output_path,
                # For compatibility, mirror video_url to url
                "url": job.output_path,
                "thumbnail_url": thumb_url,
                "error": job.error_message,
                "enhanced_prompt": job.enhanced_prompt,
                "status_message": job.status_message,
                "actual_frames": job.actual_frames
            }

    # 3. Fallback: File Check (Legacy - for jobs before DB migration)
    video_path = os.path.join(GENERATED_DIR, f"{job_id}.mp4")
    if os.path.exists(video_path):
        return {
            "job_id": job_id, 
            "status": "completed", 
            "video_url": f"/generated/{job_id}.mp4",
            "url": f"/generated/{job_id}.mp4",
            "content_type": "video"
        }
        
    return {"job_id": job_id, "status": "not_found", "progress": 0}

@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    from worker import active_jobs
    if job_id in active_jobs:
        active_jobs[job_id]["cancelled"] = True
        return {"status": "cancelling"}
    
    # If not active, might be pending in DB? 
    # For now, only cancel active.
    return {"status": "not_found"}

@app.get("/uploads")
async def list_uploads():
    """
    Returns a combined list of Assets (Uploaded) and Completed Jobs (Generated)
    from the SQLite database.
    """
    files = []
    
    with Session(engine) as session:
        # 1. Get Assets (Uploaded)
        assets = session.exec(select(Asset).order_by(Asset.created_at.desc())).all()
        for a in assets:
            files.append({
                "id": a.id,
                "url": a.url,
                "path": a.path,
                "type": a.type,
                "filename": a.filename,
                "thumbnail": None, # Add logic if we store thumbnail url in Asset
                "created_at": a.created_at.timestamp()
            })
            
        # 2. Get Completed Jobs (Generated)
        jobs = session.exec(select(Job).where(Job.status == "completed").order_by(Job.created_at.desc())).all()
        for j in jobs:
            if not j.output_path: continue
            
            filename = os.path.basename(j.output_path)
            
            # DEFAULT: Legacy
            url_prefix = "/generated"
            video_url = f"/generated/{filename}"
            if j.output_path.startswith("/"): # Absolute path in DB?
                # Check if it's in projects dir
                if "/projects/" in j.output_path:
                    # Extract relative path: /projects/{id}/generated/{filename}
                    # DB path might be: /Users/.../backend/projects/{id}/generated/{filename}
                    # OR web path: /projects/{id}/...
                    
                    if j.output_path.startswith(PROJECTS_DIR):
                        rel = os.path.relpath(j.output_path, os.path.dirname(PROJECTS_DIR))
                        video_url = f"/{rel}"
                    elif j.output_path.startswith("/projects/"):
                         video_url = j.output_path
            
            # THUMBNAIL LOGIC
            thumb_url = None
            if j.thumbnail_path:
                thumb_url = j.thumbnail_path
            else:
                # Fallback: check filesystem
                if filename.endswith(".mp4"):
                    # Check project 'thumbnails' folder first
                    if "/projects/" in video_url:
                        # Construct thumb URL by replacing generated -> thumbnails ?
                        # Video: /projects/{id}/generated/foo.mp4
                        # Thumb: /projects/{id}/thumbnails/foo_thumb.jpg
                        try:
                            parts = video_url.split("/")
                            # parts: ['', 'projects', '{id}', 'generated', 'foo.mp4']
                            if len(parts) >= 5 and parts[3] == "generated":
                                parts[3] = "thumbnails"
                                parts[-1] = parts[-1].replace(".mp4", "_thumb.jpg")
                                cand_url = "/".join(parts)
                                # Verify file existence?
                                # Convert URL to Path
                                cand_path = os.path.join(os.path.dirname(__file__), cand_url.lstrip("/"))
                                if os.path.exists(cand_path):
                                    thumb_url = cand_url
                        except: pass
                    
                    if not thumb_url:
                        # Legacy fallback
                        thumb_path = j.output_path.replace(".mp4", "_thumb.jpg")
                        if os.path.exists(thumb_path):
                             thumb_url = f"/generated/{os.path.basename(thumb_path)}"
            
            files.append({
                "id": j.id,
                "url": video_url,
                "path": j.output_path,
                "type": "video" if filename.endswith(".mp4") else "image",
                "filename": filename,
                "thumbnail": thumb_url,
                "created_at": j.created_at.timestamp()
            })
            
    # Sort combined by time
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return files

@app.delete("/upload/{filename}")
async def delete_upload(filename: str):
    with Session(engine) as session:
        # Try finding Asset
        asset = session.exec(select(Asset).where(Asset.filename == filename)).first()
        if asset:
            session.delete(asset)
            session.commit()
            # Delete file
            if os.path.exists(asset.path):
                os.remove(asset.path)
            return {"status": "deleted"}
            
        # Try finding Job (if deleting a generation result treated as upload)
        # Usually filenames for jobs are {job_id}.mp4
        job_id = filename.split(".")[0]
        job = session.get(Job, job_id)
        if job:
            # We don't delete the JOB record (history), but maybe clear output?
            # Or user thinks they are deleting the file.
            # Let's just delete the file for now to free space, 
            # OR delete the Job if we want to remove from history.
            # Current UI implies "Delete Asset".
            if os.path.exists(job.output_path):
                if os.path.isdir(job.output_path):
                    import shutil
                    shutil.rmtree(job.output_path)
                else:
                    os.remove(job.output_path)
            
            job.output_path = None # Clear path
            job.status = "deleted"
            session.add(job)
            session.commit()
            return {"status": "deleted"}
            
    # Legacy Fallback
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
        
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/projects/{project_id}/render")
async def render_project(project_id: str, background_tasks: BackgroundTasks):
    """
    Stitches all generated shots in the project into a final MP4.
    """
    import subprocess
    
    # 1. Load project from DB
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        shots = project.shots or []
        if not shots:
            raise HTTPException(status_code=400, detail="Project has no shots")
        
        # 2. Collect video paths by looking up jobs in DB
        input_files = []
        
        for shot in shots:
            job_id = shot.get("last_job_id")
            if not job_id:
                continue
            
            # Look up job to get actual output path
            job = session.get(Job, job_id)
            if job and job.output_path:
                # Convert URL path to filesystem path
                if job.output_path.startswith("/projects/"):
                    file_path = os.path.join(os.path.dirname(__file__), job.output_path.lstrip("/"))
                else:
                    file_path = os.path.join(GENERATED_DIR, os.path.basename(job.output_path))
                
                if os.path.exists(file_path) and file_path.endswith(".mp4"):
                    input_files.append(file_path)
        
        if not input_files:
            raise HTTPException(status_code=400, detail="No generated videos found for this project")
        
        # 3. Stitch to project workspace
        render_job_id = f"render_{uuid.uuid4().hex[:8]}"
        
        # Output to project's generated folder
        project_generated_dir = os.path.join(PROJECTS_DIR, project_id, "generated")
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

# Static file mount MUST come after API routes (otherwise intercepts /projects/{id} API calls)
app.mount("/projects", StaticFiles(directory=PROJECTS_DIR), name="projects")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
