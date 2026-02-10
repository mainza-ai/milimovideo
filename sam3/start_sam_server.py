import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import numpy as np
from PIL import Image
import io
import json
import logging
from contextlib import asynccontextmanager
import sys
import base64

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAM3_Server")

# Global Model Variables
sam_model = None
sam_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam_model, sam_processor
    try:
        logger.info("Loading SAM 3 Model...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)

        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Try local checkpoint first, fall back to HuggingFace download
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        local_checkpoint = os.path.join(project_root, "backend/models/sam3/sam3.pt")

        if os.path.exists(local_checkpoint):
            logger.info(f"Loading checkpoint from local: {local_checkpoint}")
            sam_model = build_sam3_image_model(
                checkpoint_path=local_checkpoint,
                device=device,
                enable_inst_interactivity=True,
                load_from_HF=False,
            )
        else:
            logger.info("Local checkpoint not found. Downloading from HuggingFace...")
            sam_model = build_sam3_image_model(
                device=device,
                enable_inst_interactivity=True,
                load_from_HF=True,
            )

        sam_model.eval()
        logger.info("SAM 3 Model loaded successfully.")

        # Initialize Sam3Processor for text/box prompts
        sam_processor = Sam3Processor(sam_model, confidence_threshold=0.5)
        logger.info("Sam3Processor initialized.")

    except Exception as e:
        logger.error(f"Failed to load SAM 3 model: {e}")
        import traceback
        traceback.print_exc()

    yield

    # Cleanup
    sam_model = None
    sam_processor = None


app = FastAPI(title="SAM 3 Microservice", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _read_image(contents: bytes) -> np.ndarray:
    """Read uploaded image bytes into a PIL Image, return numpy RGB array."""
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(pil_image)


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert a boolean/uint8 mask to PNG bytes."""
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _mask_to_base64(mask: np.ndarray) -> str:
    """Convert a mask to base64-encoded PNG string."""
    return base64.b64encode(_mask_to_png_bytes(mask)).decode("utf-8")


def _merge_masks(masks: np.ndarray) -> np.ndarray:
    """Merge multiple masks into a single binary mask via logical OR."""
    if len(masks) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    merged = masks[0].copy()
    for m in masks[1:]:
        merged = np.logical_or(merged, m)
    return merged.astype(np.uint8)


# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model_loaded": sam_model is not None,
        "processor_ready": sam_processor is not None,
    }


# ─── Point-Based Mask Prediction (Legacy) ──────────────────────────────────

@app.post("/predict/mask")
async def predict_mask(
    image: UploadFile = File(...),
    points: str = Form(...),           # JSON: [[x,y], [x,y]]
    labels: str = Form(default="[1]"), # JSON: [1, 0] (1=fg, 0=bg)
    multimask: bool = Form(False),
    boxes: str = Form(default=None),   # JSON: [cx, cy, w, h] normalized 0-1
):
    if not sam_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        np_image = _read_image(contents)

        # If box is provided, use Sam3Processor's geometric prompt
        if boxes and sam_processor:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            state = sam_processor.set_image(pil_image)

            box_coords = json.loads(boxes)
            state = sam_processor.add_geometric_prompt(
                box=box_coords, label=True, state=state
            )

            masks = state["masks"].cpu().numpy()
            scores = state["scores"].cpu().numpy()

            # Pick best mask by score
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]

            buf = io.BytesIO()
            Image.fromarray((best_mask * 255).astype(np.uint8)).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        # Standard point-based prediction
        input_points = np.array(json.loads(points))
        if len(input_points.shape) == 1:
            input_points = input_points.reshape(1, 2)
        input_labels = np.array(json.loads(labels))
        if len(input_labels.shape) == 0:
            input_labels = np.array([input_labels])

        predictor = sam_model.inst_interactive_predictor
        if predictor is None:
            raise HTTPException(status_code=500, detail="Interactive predictor not available")

        predictor.set_image(np_image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask,
        )

        if multimask:
            # Return all masks with scores as JSON
            result = []
            for i in range(len(masks)):
                result.append({
                    "mask": _mask_to_base64(masks[i]),
                    "score": float(scores[i]),
                })
            return {"masks": result}

        # Single best mask as PNG
        best_mask = masks[0]
        buf = io.BytesIO()
        Image.fromarray((best_mask * 255).astype(np.uint8)).save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Text-Prompted Object Detection ────────────────────────────────────────

@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(...),
    text: str = Form(...),
    confidence: float = Form(0.5),
):
    """
    Detect all instances of a concept described by text.
    Returns masks, bounding boxes, and confidence scores as JSON.
    """
    if not sam_processor:
        raise HTTPException(status_code=503, detail="Sam3Processor not initialized")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Set confidence threshold
        sam_processor.set_confidence_threshold(confidence)

        # Run inference
        state = sam_processor.set_image(pil_image)
        state = sam_processor.set_text_prompt(state=state, prompt=text)

        masks = state["masks"]    # [N, H, W] bool tensor
        boxes = state["boxes"]    # [N, 4] float tensor (x0, y0, x1, y1)
        scores = state["scores"]  # [N] float tensor

        if len(masks) == 0:
            return {"objects": [], "count": 0, "prompt": text}

        # Convert to JSON-serializable format
        # Squeeze channel dim: masks are [N,1,H,W] from interpolate → [N,H,W]
        masks_np = masks.squeeze(1).cpu().numpy()
        boxes_np = boxes.cpu().float().numpy()
        scores_np = scores.cpu().float().numpy()

        objects = []
        for i in range(len(masks_np)):
            objects.append({
                "id": i,
                "mask": _mask_to_base64(masks_np[i]),
                "bbox": boxes_np[i].tolist(),  # [x0, y0, x1, y1]
                "score": float(scores_np[i]),
                "label": text,
            })

        return {
            "objects": objects,
            "count": len(objects),
            "prompt": text,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Text-Prompted Segmentation (Merged Mask PNG) ──────────────────────────

@app.post("/segment/text")
async def segment_by_text(
    image: UploadFile = File(...),
    text: str = Form(...),
    confidence: float = Form(0.5),
):
    """
    Segment all instances matching the text prompt.
    Returns a single merged binary mask PNG (for direct use with inpainting).
    """
    if not sam_processor:
        raise HTTPException(status_code=503, detail="Sam3Processor not initialized")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        sam_processor.set_confidence_threshold(confidence)
        state = sam_processor.set_image(pil_image)
        state = sam_processor.set_text_prompt(state=state, prompt=text)

        masks = state["masks"]  # [N, H, W]

        if len(masks) == 0:
            # Return an all-black mask (nothing detected)
            w, h = pil_image.size
            empty = np.zeros((h, w), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(empty).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        masks_np = masks.cpu().numpy()
        merged = _merge_masks(masks_np)

        buf = io.BytesIO()
        Image.fromarray((merged * 255).astype(np.uint8)).save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text segmentation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Video Object Tracking ─────────────────────────────────────────────────

video_predictor = None

def _get_video_predictor():
    """Lazy-load the video predictor on first tracking request."""
    global video_predictor
    if video_predictor is not None:
        return video_predictor

    try:
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        logger.info("Loading SAM 3 Video Predictor (lazy init)...")

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Try local checkpoint, fall back to HF
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        local_checkpoint = os.path.join(project_root, "backend/models/sam3/sam3.pt")

        checkpoint_path = local_checkpoint if os.path.exists(local_checkpoint) else None

        video_predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            apply_temporal_disambiguation=True,
            device=device,
        )
        logger.info("SAM 3 Video Predictor loaded.")
        return video_predictor
    except Exception as e:
        logger.error(f"Failed to load video predictor: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Video predictor unavailable: {e}")


@app.post("/track/start")
async def track_start(
    video_path: str = Form(...),
    session_id: str = Form(default=None),
):
    """Start a tracking session on a video file or frame directory."""
    predictor = _get_video_predictor()

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    try:
        result = predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path,
            "session_id": session_id,
        })
        return {"session_id": result, "status": "started"}
    except Exception as e:
        logger.error(f"Track start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track/prompt")
async def track_prompt(
    session_id: str = Form(...),
    frame_idx: int = Form(0),
    text: str = Form(default=None),
    points: str = Form(default=None),        # JSON: [[x,y], ...]
    point_labels: str = Form(default=None),   # JSON: [1, 0, ...]
    boxes: str = Form(default=None),          # JSON: [[x0,y0,x1,y1], ...]
    box_labels: str = Form(default=None),     # JSON: [1, ...]
    obj_id: int = Form(default=None),
):
    """Add a prompt (text, point, or box) to a tracking session at a specific frame."""
    predictor = _get_video_predictor()

    try:
        prompt_request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_idx": frame_idx,
        }

        if text:
            prompt_request["text"] = text
        if points:
            prompt_request["points"] = json.loads(points)
        if point_labels:
            prompt_request["point_labels"] = json.loads(point_labels)
        if boxes:
            prompt_request["bounding_boxes"] = json.loads(boxes)
        if box_labels:
            prompt_request["bounding_box_labels"] = json.loads(box_labels)
        if obj_id is not None:
            prompt_request["obj_id"] = obj_id

        result = predictor.handle_request(prompt_request)
        return {"status": "prompt_added", "result": str(result)}
    except Exception as e:
        logger.error(f"Track prompt error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track/propagate")
async def track_propagate(
    session_id: str = Form(...),
    direction: str = Form("forward"),  # "forward", "backward", "both"
    start_frame: int = Form(0),
    max_frames: int = Form(default=-1),
):
    """Propagate tracking through the video. Returns results per frame as JSON."""
    predictor = _get_video_predictor()

    try:
        results = []
        for frame_result in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_idx": start_frame,
            "max_frame_num_to_track": max_frames,
        }):
            # Each frame_result is a dict with frame info and masks
            frame_data = {
                "frame_idx": frame_result.get("frame_idx", -1),
                "num_objects": frame_result.get("num_objects", 0),
            }

            # If masks are tensors, convert to base64
            if "masks" in frame_result and hasattr(frame_result["masks"], "cpu"):
                masks_np = frame_result["masks"].cpu().numpy()
                frame_data["masks"] = [_mask_to_base64(m) for m in masks_np]

            results.append(frame_data)

        return {
            "status": "complete",
            "session_id": session_id,
            "direction": direction,
            "frame_count": len(results),
            "frames": results,
        }
    except Exception as e:
        logger.error(f"Track propagate error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track/stop")
async def track_stop(session_id: str = Form(...)):
    """Close a tracking session and free GPU memory."""
    predictor = _get_video_predictor()

    try:
        predictor.handle_request({
            "type": "close_session",
            "session_id": session_id,
        })
        return {"status": "closed", "session_id": session_id}
    except Exception as e:
        logger.error(f"Track stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

