from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlmodel import Session, select
from database import get_session, Job, Shot, Project
from schemas import GenerateAdvancedRequest, GenerateImageRequest
from tasks.video import generate_video_task
from tasks.image import generate_image_task
from job_utils import active_jobs, update_job_db, update_shot_db
import uuid
import logging
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

@router.get("/status/{job_id}")
def get_status(job_id: str, session: Session = Depends(get_session)):
    # 1. Check Active Memory State (Fastest)
    if job_id in active_jobs:
        return active_jobs[job_id]
        
    # 2. Fallback to DB
    job = session.get(Job, job_id)
    if job:
        return {
            "status": job.status,
            "video_url": job.output_path, # Legacy naming
            "error": job.error_message,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "enhanced_prompt": job.enhanced_prompt,
            "status_message": job.status_message,
            "actual_frames": job.actual_frames,
            "thumbnail_url": job.thumbnail_path
        }
    
    raise HTTPException(status_code=404, detail="Job not found")

@router.get("/projects/{project_id}/active_jobs")
def get_project_active_jobs(project_id: str):
    """Returns a list of active job IDs for compliance/persistence."""
    project_jobs = []
    for j_id, meta in active_jobs.items():
        if meta.get("project_id") == project_id:
            # Return basic info to allow UI to resume polling
            project_jobs.append({
                "job_id": j_id,
                "type": meta.get("type", "unknown"),
                "status": meta.get("status"),
                "progress": meta.get("progress", 0),
                "status_message": meta.get("status_message", "")
            })
    return project_jobs

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, session: Session = Depends(get_session)):
    """Cancel a running or pending job."""
    logger.info(f"Received cancellation request for job {job_id}")
    # 1. Active Memory Cancel
    if job_id in active_jobs:
        active_jobs[job_id]["cancelled"] = True
        active_jobs[job_id]["status"] = "cancelling"
        active_jobs[job_id]["status_message"] = "Cancelling..."
        # update_job_db(job_id, "cancelled") # Don't sync DB yet; let worker see flag first!
        return {"status": "cancelling"}
    
    # 2. Pending DB Cancel (if not yet picked up by worker)
    job = session.get(Job, job_id)
    if job and job.status == "pending":
        job.status = "cancelled"
        session.add(job)
        session.commit()
        return {"status": "cancelled"}
        
    return {"status": "not_found_or_already_done"}


@router.post("/generate/advanced")
async def generate_advanced(req: GenerateAdvancedRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """
    Unified endpoint for all generation types (T2V, I2V, T2I, V2V).
    The worker inspects 'timeline' to decide the pipeline.
    """
    # Lazy import to avoid circular dep if any
    from managers.element_manager import element_manager
    import os
    import config
    
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    # Validated Params
    params = req.shot_config.model_dump()
    params["job_id"] = job_id
    params["project_id"] = req.project_id
    params["pipeline_type"] = "advanced"

    # --- Element Injection (Triggers) ---
    raw_prompt = params.get("prompt", "")
    if "@" in raw_prompt:
         enriched_prompt, element_images = element_manager.inject_elements_into_prompt(raw_prompt, req.project_id)
         params["prompt"] = enriched_prompt
         params["element_images"] = element_images
         logger.info(f"Enriched prompt: {enriched_prompt} | Visuals: {len(element_images)}")

    # --- Smart Continue / Auto-Conditioning Logic ---
    # If auto_continue is True and NO explicit image/video conditioning is provided for frame 0,
    # try to use the last completed shot from this project.
    if req.shot_config.auto_continue:
        has_initial_cond = any(
            t["frame_index"] == 0 and t["type"] in ["image", "video"] 
            for t in params.get("timeline", [])
        )
        
        if not has_initial_cond:
            # Query DB for last successful job in this project
            try:
                last_job = session.exec(
                    select(Job)
                    .where(Job.project_id == req.project_id)
                    .where(Job.status == "completed")
                    .where(Job.output_path != None)
                    .order_by(Job.created_at.desc())
                ).first()
                
                if last_job and last_job.output_path:
                    logger.info(f"Smart Continue: Extending from last job {last_job.id}")
                    
                    cond_path = last_job.output_path
                    if cond_path.endswith(".mp4") or cond_path.endswith(".mov"):
                         # We need to extract the last frame. 
                         # Project-scoped paths are REQUIRED
                         if cond_path.startswith("/projects/"):
                             # Helper to resolve FS path
                             rel = cond_path.lstrip("/projects/")
                             fs_cond_path = os.path.join(config.PROJECTS_DIR, rel)
                             
                             # Let's generate a temporary last frame path
                             project_dir = os.path.join(config.PROJECTS_DIR, req.project_id)
                             last_frame_path = os.path.join(project_dir, "thumbnails", f"{last_job.id}_auto_last.jpg")
                             
                             # Extract last frame
                             import subprocess
                             if not os.path.exists(last_frame_path):
                                 os.makedirs(os.path.dirname(last_frame_path), exist_ok=True)
                                 subprocess.run([
                                     "ffmpeg", "-y", "-sseof", "-1.0", "-i", fs_cond_path, 
                                     "-q:v", "2", "-update", "1", last_frame_path
                                 ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                             
                             cond_path = last_frame_path

                    # Add to params AND timeline
                    new_item = {
                        "path": cond_path, 
                        "frame_index": 0, 
                        "strength": 1.0, 
                        "type": "image"
                    }
                    
                    # Append to params["timeline"]
                    params["timeline"] = [new_item] + params.get("timeline", [])
                    logger.info(f"Auto-injected conditioning from {cond_path}")
                    
            except Exception as e:
                logger.error(f"Smart Continue failed: {e}")

    # Persist Job to DB (Pending)
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

# Legacy Alias
@router.post("/generate")
async def generate_video(req: GenerateAdvancedRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    return await generate_advanced(req, background_tasks, session)

@router.post("/generate/image")
def generate_image(req: GenerateImageRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    job = Job(
        id=job_id,
        type="image",
        status="pending",
        params_json=req.model_dump_json(),
        created_at=datetime.now(timezone.utc)
    )
    session.add(job)
    session.commit()
    
    params = req.model_dump()
    background_tasks.add_task(generate_image_task, job_id, params)
    
    return {"job_id": job_id, "status": "queued"}
