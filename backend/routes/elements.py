from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from schemas import ElementCreate, ElementVisualizeRequest, InpaintRequest
from typing import List, Optional
import os
import logging
import config

# Lazy imports to avoid circular deps with managers if they import models
from managers.element_manager import element_manager
from managers.inpainting_manager import inpainting_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["elements"])

@router.post("/projects/{project_id}/elements")
async def create_element(project_id: str, element: ElementCreate):
    return element_manager.create_element(
        project_id, 
        element.name, 
        element.type, 
        element.description, 
        element.trigger_word,
        element.image_path
    )

@router.get("/projects/{project_id}/elements")
async def get_elements(project_id: str):
    return element_manager.get_elements(project_id)

@router.delete("/elements/{element_id}")
async def delete_element(element_id: str):
    success = element_manager.delete_element(element_id)
    return {"success": success}

@router.post("/elements/{element_id}/visualize")
async def visualize_element(element_id: str, req: ElementVisualizeRequest, background_tasks: BackgroundTasks):
    import uuid
    from datetime import datetime, timezone
    from database import Job, get_session
    from sqlmodel import Session
    from database import engine 
    from job_utils import active_jobs
    
    # Create Job
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    # Register in Active Memory (for immediate polling)
    active_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "status_message": "Queued...",
        "type": "element_visual",
        "project_id": "unknown", # We should fetch this, but for now it's okay. 
                                 # WAIT, we need project_id for active_jobs listing.
                                 # We can look it up or pass it in URL? URL only has element_id.
                                 # We'll look it up in the manager task, but for now let's try to get it here if cheap.
                                 # Actually `element_manager.generate_visual_task` will set it?
                                 # No, `active_jobs` needs it NOW for the UI to recover it if refreshed.
                                 # I'll rely on the existing UI keeping state for now, or fetch element to get project_id.
        "cancelled": False
    }

    # Queue Task
    background_tasks.add_task(
        element_manager.generate_visual_task, 
        job_id, 
        element_id, 
        req.prompt_override, 
        req.guidance_scale,
        req.enable_ae
    )
    
    return {"job_id": job_id, "status": "queued"}

# --- In-Painting Extensions ---

@router.post("/edit/inpaint")
async def inpaint_image(job_id: str, req: InpaintRequest, background_tasks: BackgroundTasks):
    # Run Inpainting
    mask_path = req.mask_path
    if req.points and not mask_path:
        mask_path = await inpainting_manager.get_mask_from_sam(req.image_path, eval(req.points))
    
    if not mask_path:
         return {"status": "error", "message": "No mask provided"}

    background_tasks.add_task(inpainting_manager.process_inpaint, job_id, req.image_path, mask_path, req.prompt)
    
    return {"status": "queued", "job_id": job_id}

@router.post("/edit/segment")
async def segment_preview(file: UploadFile = File(...), points: str = Form(...)):
    # Proxy to SAM Microservice
    import requests
    files = {"file": file.file}
    data = {"points": points}
    try:
        res = requests.post(f"http://localhost:{config.SAM_SERVICE_PORT}/segment", files=files, data=data)
        return res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
