from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Request
from schemas import ElementCreate, ElementUpdate, ElementVisualizeRequest, InpaintRequest, TrackingSaveRequest
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

@router.put("/elements/{element_id}")
async def update_element(element_id: str, update: ElementUpdate):
    """Update an element's properties (name, description, trigger_word, etc.)."""
    updates = update.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = element_manager.update_element(element_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found")
    return result

# --- SAM Service Health Check ---

@router.get("/sam/health")
async def sam_health_check():
    """Check if the SAM 3 microservice is reachable."""
    import requests as http_requests
    import time
    try:
        start = time.time()
        res = http_requests.get(
            f"http://localhost:{config.SAM_SERVICE_PORT}/health",
            timeout=3
        )
        latency_ms = round((time.time() - start) * 1000)
        return {
            "online": res.status_code == 200,
            "latency_ms": latency_ms,
            "status": res.json() if res.status_code == 200 else None
        }
    except Exception:
        return {"online": False, "latency_ms": -1, "status": None}

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

class TrackRemoveRequest(BaseModel):
    session_id: str
    obj_id: int


@router.post("/track/start")
async def track_start(req: TrackStartRequest):
    """Start a SAM video tracking session."""
    video_path = req.video_path

    # Resolve URL paths to absolute filesystem paths
    # Frontend sends paths like "/projects/{id}/generated/job_xxx.mp4"
    # or full URLs like "http://localhost:8000/projects/..."
    if video_path.startswith("http"):
        from urllib.parse import urlparse
        video_path = urlparse(video_path).path  # Strip scheme+host

    if video_path.startswith("/projects/"):
        video_path = os.path.join(config.PROJECTS_DIR, video_path[len("/projects/"):])
    elif video_path.startswith("/generated/"):
        legacy_generated = os.path.join(config.BACKEND_DIR, "generated")
        video_path = os.path.join(legacy_generated, video_path[len("/generated/"):])
    elif video_path.startswith("/uploads/"):
        legacy_uploads = os.path.join(config.BACKEND_DIR, "uploads")
        video_path = os.path.join(legacy_uploads, video_path[len("/uploads/"):])

    video_path = os.path.normpath(video_path)
    logger.info(f"Track start — resolved video path: {video_path}")

    result = tracking_manager.start_session(video_path, req.session_id)
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


@router.post("/track/remove")
async def track_remove_object(req: TrackRemoveRequest):
    """Remove an object from the tracking session."""
    result = tracking_manager.remove_object(req.session_id, req.obj_id)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@router.post("/edit/track/save")
async def save_tracking_results(request: TrackingSaveRequest):
    """Save tracking results (masks) to disk to prevent data loss."""
    try:
        import base64
        import json
        import hashlib
        from PIL import Image
        import io

        # Use video_path hash as persistence key (session IDs are ephemeral)
        video_path = getattr(request, 'video_path', None) or request.session_id
        path_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
        session_dir = os.path.join(config.BACKEND_DIR, "exports", "tracking", path_hash)
        masks_dir = os.path.join(session_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        saved_files = []
        frame_map = {}

        for frame in request.frames:
            frame_idx = frame.get("frame_idx", -1)
            masks = frame.get("masks", {})
            scores = frame.get("scores", {})
            
            frame_map[frame_idx] = {
                "num_objects": frame.get("num_objects", 0),
                "objects": []
            }

            for obj_id_str, b64_mask in masks.items():
                try:
                    # Decode base64
                    img_data = base64.b64decode(b64_mask)
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Save as PNG: frame_XXXXX_obj_Y.png
                    filename = f"frame_{int(frame_idx):05d}_obj_{obj_id_str}.png"
                    filepath = os.path.join(masks_dir, filename)
                    img.save(filepath, "PNG")
                    
                    saved_files.append(filepath)
                    
                    # Metadata
                    frame_map[frame_idx]["objects"].append({
                        "id": obj_id_str,
                        "score": scores.get(obj_id_str, 0.0),
                        "file": filename
                    })
                except Exception as e:
                    logger.error(f"Failed to save mask for frame {frame_idx} obj {obj_id_str}: {e}")

        # Save manifest
        manifest_path = os.path.join(session_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            manifest_data = {
                "session_id": request.session_id,
                "video_path": video_path,
                "total_frames": len(request.frames),
                "frames": frame_map,
            }
            if request.objects:
                 manifest_data["objects"] = request.objects
            
            json.dump(manifest_data, f, indent=2)

        return {
            "status": "saved",
            "path": session_dir,
            "count": len(saved_files)
        }
    except Exception as e:
        logger.error(f"Tracking save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/track/load")
async def load_tracking_results(request: Request):
    """Load tracking results (masks) from disk."""
    try:
        import base64
        import json
        import hashlib

        body = await request.json()
        video_path = body.get('video_path', '')
        session_id = body.get('session_id', '')
        
        # Try video_path hash first, fall back to session_id
        path_hash = hashlib.md5(video_path.encode()).hexdigest()[:12] if video_path else ''
        candidates = []
        if path_hash:
            candidates.append(os.path.join(config.BACKEND_DIR, "exports", "tracking", path_hash))
        if session_id:
            candidates.append(os.path.join(config.BACKEND_DIR, "exports", "tracking", session_id))
        
        session_dir = None
        manifest_path = None
        for d in candidates:
            mp = os.path.join(d, "manifest.json")
            if os.path.exists(mp):
                session_dir = d
                manifest_path = mp
                break
        
        if not manifest_path:
             return {"status": "not_found", "frames": []}

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        frames = []
        # Reconstruct frameResults format
        # manifest["frames"] is a dict: frame_idx -> { objects: [...] }
        for frame_idx_str, data in manifest.get("frames", {}).items():
            masks = {}
            scores = {}
            for obj in data.get("objects", []):
                obj_id = obj["id"]
                score = obj["score"]
                filename = obj["file"]
                filepath = os.path.join(session_dir, "masks", filename)
                
                if os.path.exists(filepath):
                    with open(filepath, "rb") as img_f:
                        b64_mask = base64.b64encode(img_f.read()).decode('utf-8')
                        masks[obj_id] = b64_mask
                        scores[obj_id] = score
            
            if masks:
                frames.append({
                    "frame_idx": int(frame_idx_str),
                    "masks": masks,
                    "scores": scores,
                    "num_objects": len(masks)
                })

        return {
            "status": "loaded",
            "frames": frames,
            "session_id": session_id,
            "objects": manifest.get("objects", {})
        }

    except Exception as e:
        logger.error(f"Tracking load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
