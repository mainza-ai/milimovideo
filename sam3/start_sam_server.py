import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
import logging
from contextlib import asynccontextmanager
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAM3_Server")

# Global Model Variable
sam_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam_model
    try:
        logger.info("Loading SAM 3 Model...")
        # Add parent directory to path to ensure we can import sam3 package if needed
        # Assuming run from root: python sam3/start_sam_server.py
        # Or if run from inside sam3 dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        # Also need to make sure 'sam3' package is importable. 
        # It seems 'sam3' is a subdirectory in the repo root 'sam3/sam3'.
        # We might need to adjust based on where 'sam3' package really is.
        # Based on file structure: milestones/sam3/sam3 exists.
        
        # Import inside startup to avoid import errors if not installed
        try:
             from sam3 import build_sam3_image_model as build_sam3
        except ImportError:
             # Try appending parent dir if run from root
             parent_dir = os.path.dirname(current_dir)
             if parent_dir not in sys.path:
                 sys.path.append(parent_dir)
             from sam3 import build_sam3_image_model as build_sam3
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Paths - Update to point to backend/models/sam3.pt
        # We need absolute path or relative to project root
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        checkpoint = os.path.join(project_root, "backend/models/sam3.pt")
        
        logger.info(f"Loading checkpoint from: {checkpoint}")
        
        # For now, we'll try to load if the file exists, otherwise warn
        if os.path.exists(checkpoint):
            # Enable instance interactivity to get the predictor
            sam_model = build_sam3(
                checkpoint_path=checkpoint, 
                device=device,
                enable_inst_interactivity=True
            ).to(device)
            sam_model.eval() # Set to eval mode
            logger.info("SAM 3 Model Loaded Successfully.")
        else:
            logger.warning(f"Checkpoint {checkpoint} not found. Model not loaded.")

    except Exception as e:
        logger.error(f"Failed to load SAM 3 model: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Cleanup if needed
    sam_model = None

app = FastAPI(title="SAM 3 Microservice", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "running", "model_loaded": sam_model is not None}

@app.post("/predict/mask")
async def predict_mask(
    image: UploadFile = File(...),
    points: str = Form(...), # JSON string of points: [[x,y], [x,y]]
    labels: str = Form(default="[1]"), # JSON string of labels: [1, 0] (1=foreground, 0=background)
    multimask: bool = Form(False)
):
    if not sam_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read Image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        np_image = np.array(pil_image)

        # Parse Points/Labels
        import json
        try:
            input_points = np.array(json.loads(points))
            # Handle list of lists normalization if needed
            if len(input_points.shape) == 1:
                input_points = input_points.reshape(1, 2)
                
            input_labels = np.array(json.loads(labels))
            if len(input_labels.shape) == 0:
                 input_labels = np.array([input_labels])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid points/labels format: {e}")

        # Prediction Logic
        # Use the predictor attached to the model
        predictor = sam_model.inst_interactive_predictor
        if predictor is None:
             raise HTTPException(status_code=500, detail="Interactive predictor not available in model")
        
        # SAM 3 usually expects a batch dimension or specific call structure?
        # Based on typical SAM usage: predictor.set_image(image) -> predictor.predict(...)
        # The 'inst_interactive_predictor' might be a wrapper.
        
        predictor.set_image(np_image)

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask,
        )

        # Return mask
        # If multimask is False, we just return the best mask (which is usually the first one or we can pick based on scores)
        # The predictor returns masks as [N, H, W] where N is number of masks (1 or 3)
        # We usually want the first one if multimask is disabled.
        
        # We need to return the mask as a binary image (for frontend/backend usage) OR raw bytes?
        # Returning a PNG image is usually easier for consumers than pure JSON arrays for large masks.
        
        best_mask = masks[0]
        # Convert to PIL Image (0 or 255)
        mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))
        
        # Save to buffer
        buf = io.BytesIO()
        mask_image.save(buf, format="PNG")
        buf.seek(0)
        
        # Return as streaming response
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
