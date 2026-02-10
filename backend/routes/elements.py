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
        "project_id": "unknown",
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

# --- Editing & SAM Integration ---

@router.post("/edit/inpaint")
async def inpaint_image(job_id: str, req: InpaintRequest, background_tasks: BackgroundTasks):
    """Run Flux inpainting. Mask can come from: mask_path, SAM points, or SAM text prompt."""
    from job_utils import active_jobs
    from sqlmodel import Session as DBSession
    from database import engine as db_engine, Job
    from datetime import datetime, timezone

    mask_path = req.mask_path

    # Priority: explicit mask > text-based SAM > point-based SAM
    if not mask_path and req.text_mask:
        mask_path = await inpainting_manager.get_mask_from_text(req.image_path, req.text_mask)
    elif not mask_path and req.points:
        mask_path = await inpainting_manager.get_mask_from_sam(req.image_path, eval(req.points))

    if not mask_path:
        return {"status": "error", "message": "No mask provided or generated"}

    # Extract project_id from image path: .../projects/{project_id}/...
    project_id = None
    try:
        rel = req.image_path.replace(config.PROJECTS_DIR, "").strip(os.sep)
        parts = rel.split(os.sep)
        if parts:
            project_id = parts[0]
    except Exception:
        pass

    # Register in active_jobs for status polling + SSE progress
    active_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "status_message": "In-Painting...",
        "type": "inpaint",
        "cancelled": False,
        "project_id": project_id
    }

    # Persist Job record to DB so /status/{job_id} works after active_jobs cleanup
    try:
        with DBSession(db_engine) as session:
            job = Job(
                id=job_id,
                project_id=project_id,
                type="inpaint",
                status="pending",
                prompt=req.prompt,
                created_at=datetime.now(timezone.utc),
                status_message="In-Painting..."
            )
            session.add(job)
            session.commit()
    except Exception as e:
        logger.error(f"Failed to persist inpaint Job record: {e}")

    background_tasks.add_task(inpainting_manager.process_inpaint, job_id, req.image_path, mask_path, req.prompt)

    return {"status": "queued", "job_id": job_id, "mask_path": mask_path}


@router.post("/edit/segment")
async def segment_preview(
    image: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(default="[1]"),
    multimask: bool = Form(False),
):
    """Point-based segmentation preview. Proxies to SAM /predict/mask."""
    import requests as http_requests
    try:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"points": points, "labels": labels, "multimask": str(multimask).lower()}
        res = http_requests.post(
            f"http://localhost:{config.SAM_SERVICE_PORT}/predict/mask",
            files=files, data=data
        )
        if res.status_code == 200:
            from fastapi.responses import StreamingResponse
            import io
            return StreamingResponse(io.BytesIO(res.content), media_type="image/png")
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/detect")
async def detect_objects(
    image: UploadFile = File(...),
    text: str = Form(...),
    confidence: float = Form(0.5),
):
    """Text-based multi-object detection. Proxies to SAM /detect."""
    import requests as http_requests
    try:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"text": text, "confidence": str(confidence)}
        res = http_requests.post(
            f"http://localhost:{config.SAM_SERVICE_PORT}/detect",
            files=files, data=data
        )
        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detect proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/segment-text")
async def segment_by_text(
    image: UploadFile = File(...),
    text: str = Form(...),
    confidence: float = Form(0.5),
):
    """Text-based segmentation. Proxies to SAM /segment/text, returns merged mask PNG."""
    import requests as http_requests
    try:
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"text": text, "confidence": str(confidence)}
        res = http_requests.post(
            f"http://localhost:{config.SAM_SERVICE_PORT}/segment/text",
            files=files, data=data
        )
        if res.status_code == 200:
            from fastapi.responses import StreamingResponse
            import io
            return StreamingResponse(io.BytesIO(res.content), media_type="image/png")
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segment-text proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Video Object Tracking ---

from managers.tracking_manager import tracking_manager
from pydantic import BaseModel
from typing import Optional as OptType

class TrackStartRequest(BaseModel):
    video_path: str
    session_id: OptType[str] = None

class TrackPromptRequest(BaseModel):
    session_id: str
    frame_idx: int = 0
    text: OptType[str] = None
    points: OptType[list] = None
    point_labels: OptType[list] = None
    boxes: OptType[list] = None
    box_labels: OptType[list] = None
    obj_id: OptType[int] = None

class TrackPropagateRequest(BaseModel):
    session_id: str
    direction: str = "forward"
    start_frame: int = 0
    max_frames: int = -1

class TrackStopRequest(BaseModel):
    session_id: str


@router.post("/track/start")
async def track_start(req: TrackStartRequest):
    """Start a SAM video tracking session."""
    result = tracking_manager.start_session(req.video_path, req.session_id)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/track/prompt")
async def track_add_prompt(req: TrackPromptRequest):
    """Add a text/point/box prompt to a tracking session."""
    result = tracking_manager.add_prompt(
        session_id=req.session_id,
        frame_idx=req.frame_idx,
        text=req.text,
        points=req.points,
        point_labels=req.point_labels,
        boxes=req.boxes,
        box_labels=req.box_labels,
        obj_id=req.obj_id,
    )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/track/propagate")
async def track_propagate(req: TrackPropagateRequest):
    """Propagate tracking across video frames."""
    result = tracking_manager.propagate(
        session_id=req.session_id,
        direction=req.direction,
        start_frame=req.start_frame,
        max_frames=req.max_frames,
    )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/track/stop")
async def track_stop(req: TrackStopRequest):
    """Stop a tracking session and free resources."""
    result = tracking_manager.stop_session(req.session_id)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

