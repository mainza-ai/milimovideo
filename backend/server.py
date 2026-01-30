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

# Ensure LTX-2 packages are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../LTX-2/packages/ltx-core/src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../LTX-2/packages/ltx-pipelines/src")))

from worker import generate_video_task

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "projects")

from events import event_manager
from database import init_db, get_session, Project, Job, Asset, engine
from sqlmodel import Session, select

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Backend V2 starting...")
    init_db() # Initialize SQLite DB
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    
    # Sync filesystem to DB (Recover history)
    try:
        sync_storage()
    except Exception as e:
        logger.error(f"Sync storage failed: {e}")
        
    yield
    print("Backend V2 shutting down...")

def sync_storage():
    """
    Scans UPLOAD_DIR and GENERATED_DIR to ensure all files are indexed in the DB.
    Useful for migration or if files are added manually.
    """
    logger.info("Syncing storage with database...")
    with Session(engine) as session:
        # 1. Sync Uploads
        upload_files = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith(".")]
        for filename in upload_files:
            if "_thumb" in filename: continue
            
            # Check if exists
            exists = session.exec(select(Asset).where(Asset.filename == filename)).first()
            if not exists:
                path = os.path.join(UPLOAD_DIR, filename)
                ext = filename.split(".")[-1].lower()
                
                # Try to use file creation time
                try:
                    ctime = datetime.fromtimestamp(os.path.getctime(path), tz=timezone.utc)
                except:
                    ctime = datetime.now(timezone.utc)
                    
                asset = Asset(
                    id=filename.split(".")[0], # Best guess ID
                    type="video" if ext in ["mp4", "mov"] else "image",
                    path=path,
                    url=f"/uploads/{filename}",
                    filename=filename,
                    created_at=ctime
                )
                session.add(asset)
                logger.info(f"Indexed orphan upload: {filename}")
        
        # 2. Sync Generated
        gen_files = [f for f in os.listdir(GENERATED_DIR) if not f.startswith(".")]
        for filename in gen_files:
            if "_thumb" in filename or "_list" in filename or "_last" in filename or "_part" in filename: continue
            
            path = os.path.join(GENERATED_DIR, filename)
            # Check if linked to any job
            # This is harder because job.output_path is the key, but it might be absolute or relative?
            # We stored absolute path in recent code.
            
            # Simple check: does a job exist with this ID?
            job_id = filename.split(".")[0]
            exists = session.get(Job, job_id)
            
            if not exists:
                # Create a legacy job record
                try:
                    ctime = datetime.fromtimestamp(os.path.getctime(path), tz=timezone.utc)
                except:
                    ctime = datetime.now(timezone.utc)
                
                job = Job(
                    id=job_id,
                    type="unknown", # Legacy
                    status="completed",
                    created_at=ctime,
                    completed_at=ctime,
                    output_path=path,
                    prompt="Legacy File (Restored)"
                )
                session.add(job)
                logger.info(f"Indexed orphan generation: {filename}")
                
        session.commit()
    logger.info("Storage sync complete.")

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

app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

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
async def upload_asset(file: UploadFile = File(...)):
    """Uploads an asset and returns a permanent path."""
    ext = file.filename.split(".")[-1].lower()
    asset_id = uuid.uuid4().hex
    filename = f"{asset_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    item_type = "video" if ext in ["mp4", "mov", "avi"] else "image"
    thumb_url = None
    
    if item_type == "video":
        thumb_path = generate_thumbnail(file_path)
        if thumb_path:
            thumb_filename = os.path.basename(thumb_path)
            thumb_url = f"/uploads/{thumb_filename}"
            
    # Persist to DB
    with Session(engine) as session:
        asset = Asset(
            id=asset_id,
            type=item_type,
            path=file_path,
            url=f"/uploads/{filename}",
            filename=file.filename,
            created_at=datetime.now(timezone.utc)
        )
        session.add(asset)
        session.commit()
            
    return {
        "asset_id": asset_id,
        "url": f"/uploads/{filename}",
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
    # 1. Check if video exists
    video_path = os.path.join(GENERATED_DIR, f"{job_id}.mp4")
    
    # Check for image if video doesn't exist
    if not os.path.exists(video_path):
        image_path = os.path.join(GENERATED_DIR, f"{job_id}.jpg")
        if os.path.exists(image_path):
             return {
                "url": f"/generated/{job_id}.jpg",
                "path": image_path
            }
        
        # Also check for _last.jpg from previous frame extractions?
        # Or if the user generated a single frame, the job output is .jpg.
            
        raise HTTPException(status_code=404, detail="Video/Image not found")
        
    try:
        # 2. Extract last frame
        out_name = f"{job_id}_last.jpg"
        out_path = os.path.join(GENERATED_DIR, out_name)
        
        # Use ffmpeg to get last frame
        # -sseof -0.1 gets the last 0.1s
        import subprocess
        # Robust Last Frame Extraction:
        # 1. Seek to last 1.0s (-sseof -1.0) to ensure we catch the stream even if duration metadata is slightly off.
        # 2. Output all frames in that window, overwriting the file (-update 1).
        # 3. The final result on disk is the absolute last frame.
        subprocess.run([
             "ffmpeg", "-y", "-sseof", "-1.0", "-i", video_path, 
             "-q:v", "2", "-update", "1", out_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return {
            "url": f"/generated/{out_name}",
            "path": out_path
        }
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
    
    return {"job_id": job_id, "status": "queued"}

# --- Project Persistence ---

@app.post("/project")
async def create_project(request: CreateProjectRequest):
    project_id = uuid.uuid4().hex
    
    with Session(engine) as session:
        db_project = Project(
            id=project_id,
            name=request.name,
            shots=[],
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
            "shots": db_project.shots,
            "seed": db_project.seed,
            "resolution_w": db_project.resolution_w,
            "resolution_h": db_project.resolution_h,
            "fps": db_project.fps
        }

@app.get("/project/{project_id}")
async def get_project(project_id: str):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        
        if not project:
            # Fallback: Check if it exists as legacy JSON but not in DB? 
            # (Unlikely given create_project history, but possible if DB was wiped but files kept)
            legacy_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
            if os.path.exists(legacy_path):
                # Restore to DB
                with open(legacy_path, "r") as f:
                    data = json.load(f)
                
                restored_project = Project(
                    id=data.get("id", project_id),
                    name=data.get("name", "Restored Project"),
                    shots=data.get("shots", []),
                    resolution_w=data.get("resolution_w", 768),
                    resolution_h=data.get("resolution_h", 512),
                    fps=data.get("fps", 25),
                    seed=data.get("seed", 42)
                )
                session.add(restored_project)
                session.commit()
                return data
            
            raise HTTPException(status_code=404, detail="Project not found")

        # Lazy Migration: If DB has no shots but JSON exists (Migration scenario)
        if not project.shots:
            legacy_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
            if os.path.exists(legacy_path):
                try:
                    with open(legacy_path, "r") as f:
                        data = json.load(f)
                        if data.get("shots"):
                            project.shots = data["shots"]
                            # Also update settings if they were default in DB
                            project.resolution_w = data.get("resolution_w", project.resolution_w)
                            project.resolution_h = data.get("resolution_h", project.resolution_h)
                            project.fps = data.get("fps", project.fps)
                            project.seed = data.get("seed", project.seed)
                            
                            session.add(project)
                            session.commit()
                            logger.info(f"Lazily migrated project {project_id} from JSON to DB")
                except Exception as e:
                    logger.error(f"Failed to migrate project {project_id}: {e}")

        # Lazy Migration: If DB has no shots but JSON exists
        if not project.shots:
             # ... (existing JSON migration code) ...
             pass # Logic remains from previous block, just ensuring we don't break it.

        # --- AUTO-HYDRATION (Recover "lost" shots) ---
        # Fetch all completed jobs for this project
        completed_jobs = session.exec(
            select(Job)
            .where(Job.project_id == project_id)
            .where(Job.status == "completed")
            .where(Job.output_path != None)
            .order_by(Job.created_at.asc())
        ).all()
        
        logger.info(f"[Hydration] Project {project_id}: Found {len(completed_jobs)} completed jobs in DB.")

        current_shot_ids = {s.get("id") for s in project.shots}
        logger.info(f"[Hydration] Current Project Shots: {len(project.shots)} IDs: {list(current_shot_ids)}")
        
        shots_restored = False

        for job in completed_jobs:
            # Determine Shot ID (either from params or Job ID if missing)
            try:
                params = json.loads(job.params_json) if job.params_json else {}
                shot_id = params.get("id", job.id)
            except:
                shot_id = job.id
            
            # Check if this shot is missing from the Project Timeline
            if shot_id not in current_shot_ids:
                logger.info(f"[Hydration] RESTORING missing shot {shot_id} from Job {job.id}")
                
                # Reconstruct Shot Object (snake_case for DB/Frontend)
                reconstructed_shot = {
                    "id": shot_id,
                    "prompt": job.prompt or "Recovered Shot",
                    "negative_prompt": "", 
                    "seed": params.get("seed", 42),
                    "width": params.get("width", project.resolution_w),
                    "height": params.get("height", project.resolution_h),
                    "num_frames": job.actual_frames or params.get("num_frames", 121),
                    # "fps": params.get("fps", project.fps), 
                    "timeline": [], 
                    "cfg_scale": params.get("cfg_scale", 3.0),
                    "enhance_prompt": True,
                    "upscale": True,
                    "pipeline_override": "auto",
                    "last_job_id": job.id,
                    # We send relative path in URL, frontend handles full URL conversion? 
                    # Actually job.output_path in DB is usually /generated/....mp4 (absolute string but relative to domain)
                    # Frontend expects snake_case keys.
                    # video_url is not strictly needed if last_job_id is present, but good for cache.
                    # "video_url": ...
                    "thumbnail_url": job.thumbnail_path, 
                    "enhanced_prompt_result": job.enhanced_prompt
                }
                
                # Add to project
                project.shots.append(reconstructed_shot)
                current_shot_ids.add(shot_id)
                shots_restored = True
            else:
                 logger.info(f"[Hydration] Skip {shot_id}, already exists.")
        
        # --- SYNC EXISTING SHOTS (Update Metadata from Job Truth) ---
        shots_updated = False
        job_map = {j.id: j for j in completed_jobs}
        
        # We must use a separate list or careful iteration if we plan to modify
        for i, shot in enumerate(project.shots):
             matched_job = None
             # Check snake_case keys first (DB format)
             if shot.get("last_job_id") and shot["last_job_id"] in job_map:
                 matched_job = job_map[shot["last_job_id"]]
             elif shot.get("id") in job_map:
                 matched_job = job_map[shot["id"]]
             
             if matched_job:
                 changes = False
                 # Sync Thumbnail
                 if matched_job.thumbnail_path and shot.get("thumbnail_url") != matched_job.thumbnail_path:
                     shot["thumbnail_url"] = matched_job.thumbnail_path
                     changes = True
                 
                 # Sync Frames
                 if matched_job.actual_frames and shot.get("num_frames") != matched_job.actual_frames:
                     shot["num_frames"] = matched_job.actual_frames
                     changes = True
                 
                 # Sync Video URL (if we store it in DB shots)
                 # Frontend usually generates it from last_job_id, but if we store explicit video_url:
                 # It's usually not stored in DB shots by default unless we saved it.
                 # But let's actally NOT force video_url if it's dynamic.
                 # Wait, if we are restoring, we probably want to ensure it's playable.
                 # But frontend `loadProject` does `videoUrl: s.last_job_id ? getAssetUrl(...) : undefined`.
                 # So `last_job_id` is the key.
                 
                 # Sync enhanced prompt
                 if matched_job.enhanced_prompt and shot.get("enhanced_prompt_result") != matched_job.enhanced_prompt:
                     shot["enhanced_prompt_result"] = matched_job.enhanced_prompt
                     changes = True
                 
                 if changes:
                     project.shots[i] = shot
                     shots_updated = True
                     logger.info(f"[Hydration] Synced shot {shot.get('id')} with latest Job data (snake_case)")

        if shots_restored or shots_updated:
            # Persist immediately so it sticks
            # Re-assign to trigger SQLModel/SQLAlchemy change detection for JSON column
            project.shots = list(project.shots) 
            session.add(project)
            session.commit()
            session.refresh(project)
            logger.info(f"[Hydration] SUCCESS. Project {project_id} saved with {len(project.shots)} shots.")

        # Construct response matching Frontend Project interface
        return {
            "id": project.id,
            "name": project.name,
            "shots": project.shots,
            "settings": { 
                "resolution_w": project.resolution_w,
                "resolution_h": project.resolution_h,
                "fps": project.fps
            },
            "fps": project.fps, 
            "resolution_w": project.resolution_w,
            "resolution_h": project.resolution_h,
            "seed": project.seed,
            "_debug_jobs_found": len(completed_jobs),
            "_debug_restored": shots_restored
        }

@app.put("/project/{project_id}")
async def save_project(project_id: str, state: ProjectState):
    if state.id != project_id:
        raise HTTPException(status_code=400, detail="Project ID mismatch")
    
    with Session(engine) as session:
        db_project = session.get(Project, project_id)
        if not db_project:
             # Create if missing?
             raise HTTPException(status_code=404, detail="Project not found")
        
        db_project.name = state.name
        db_project.shots = state.shots
        
        # Save settings
        db_project.resolution_w = state.resolution_w
        db_project.resolution_h = state.resolution_h
        db_project.fps = state.fps
        db_project.seed = state.seed
        
        db_project.updated_at = datetime.now(timezone.utc)
        
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

@app.delete("/project/{project_id}")
async def delete_project(project_id: str):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
            
        session.delete(project)
        session.commit()
        
        # Delete JSON file if exists
        path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
        if os.path.exists(path):
            os.remove(path)
            
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
                "video_url": f"/generated/{os.path.basename(job.output_path)}" if job.output_path else None,
                # For compatibility, mirror video_url to url
                "url": f"/generated/{os.path.basename(job.output_path)}" if job.output_path else None,
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
            url_prefix = "/generated"
            
            # Simple thumb check
            thumb_url = None
            if filename.endswith(".mp4"):
                thumb_path = j.output_path.replace(".mp4", "_thumb.jpg")
                if os.path.exists(thumb_path):
                    thumb_url = f"{url_prefix}/{os.path.basename(thumb_path)}"
            
            files.append({
                "id": j.id,
                "url": f"{url_prefix}/{filename}",
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

@app.post("/project/{project_id}/render")
async def render_project(project_id: str, background_tasks: BackgroundTasks):
    """
    Stitches all generated shots in the project into a final MP4.
    """
    # 1. Load project
    project_path = os.path.join(PROJECTS_DIR, f"{project_id}.json")
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
        
    with open(project_path, "r") as f:
        project_data = json.load(f)
        
    shots = project_data.get("shots", [])
    if not shots:
        raise HTTPException(status_code=400, detail="Project has no shots")
        
    # 2. Collect video paths
    # We assume the frontend checks if all shots are 'done' before calling render,
    # or we construct the path deterministically from shot ID?
    # Actually, the 'ShotConfig' has an ID. The worker saves output as what?
    # Worker currently saves as `job_id.mp4`.
    # The frontend needs to link Shot -> Job ID. 
    # For now, let's assume the ShotConfig in the project JSON *has* a 'result_asset_id' or 'job_id' field 
    # that gets updated by the frontend when a job completes.
    
    # We'll update the ShotConfig model slightly for this context, but since it's a dict here:
    input_files = []
    
    for shot in shots:
        # Check if shot has a result
        job_id = shot.get("last_job_id")
        if not job_id:
             # Skip or error? Let's skip validly to allow partial renders?
             continue
             
        file_path = os.path.join(GENERATED_DIR, f"{job_id}.mp4")
        if os.path.exists(file_path):
            input_files.append(file_path)
            
    if not input_files:
        raise HTTPException(status_code=400, detail="No generated videos found for this project")
        
    # 3. Stitch
    render_job_id = f"render_{uuid.uuid4().hex[:8]}"
    output_path = os.path.join(GENERATED_DIR, f"{render_job_id}.mp4")
    
    # We run this in background or sync? Stitching is fast-ish but safer in background.
    # But user wants a URL back.
    # Let's do a sync wait if short, or background. FFMPEG concat is fast.
    # Let's run it here.
    
    try:
        list_file_path = os.path.join(GENERATED_DIR, f"{render_job_id}_list.txt")
        with open(list_file_path, "w") as f:
            for mp4 in input_files:
                f.write(f"file '{mp4}'\n")
        
        import subprocess
        # ffmpeg concat
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path,
            "-c", "copy", "-y", output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Cleanup list
        if os.path.exists(list_file_path):
            os.remove(list_file_path)
            
        return {"status": "completed", "video_url": f"/generated/{render_job_id}.mp4"}
        
    except Exception as e:
        print(f"Render failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
