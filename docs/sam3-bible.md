# SAM 3 Repository Documentation

## 1. Executive Summary

SAM 3 (Segment Anything Model 3) is Meta's state-of-the-art segmentation system, designed to provide **promptable, zero-shot segmentation** of arbitrary objects in images and video. Given an image and one or more point prompts (clicks indicating foreground/background), SAM 3 produces a high-quality binary mask of the selected object.

SAM 3 extends its predecessor (SAM 2) with:
- **Improved Concept Understanding**: Better at segmenting ambiguous or complex objects.
- **Agentic Workflow Support**: Designed for integration into automated pipelines.
- **Instance-Level Interactivity**: Built-in predictor for interactive segmentation.

Milimo integrates SAM 3 as a **standalone microservice** running on its own port, called by the main backend for mask generation during inpainting workflows.

## 2. System Architecture Overview

### Architectural Style
SAM 3 uses an encoder-decoder architecture:

1.  **Image Encoder**: A Vision Transformer (ViT) that processes the input image into dense feature embeddings.
2.  **Prompt Encoder**: Encodes sparse prompts (points, boxes, text) into the same embedding space.
3.  **Mask Decoder**: A lightweight transformer that takes encoded image features + prompt embeddings and outputs segmentation masks.

### Key Capabilities
- **Zero-Shot**: Segments objects it was never specifically trained on.
- **Multi-Mask Output**: Can return multiple candidate masks ranked by confidence.
- **Point Prompting**: Single or multiple foreground/background clicks.
- **Box Prompting**: Bounding box around target region.
- **Instance Interactivity**: The `inst_interactive_predictor` provides a session-based API where the image is set once, then multiple prediction calls share cached features.

## 3. Core Concepts

- **Promptable Segmentation**: Unlike traditional semantic segmentation (which requires fixed classes), SAM takes spatial prompts (points/boxes) and segments whatever is at that location.
- **Foreground/Background Labels**: Points are labeled `1` (foreground = include) or `0` (background = exclude). Multiple points can refine the selection.
- **Multi-Mask vs Single-Mask**: When `multimask_output=True`, SAM returns 3 candidate masks (whole object, part, subpart). When `False`, returns the single best mask.

## 4. Model Architecture Details

| Component | Details |
|---|---|
| **Image Encoder** | Vision Transformer (ViT-H or similar variant) |
| **Mask Decoder** | Lightweight transformer decoder |
| **Parameters** | ~636M (base), varies by variant |
| **Input** | RGB image (any resolution, internally resized) |
| **Output** | Binary mask(s) `[N, H, W]` where N ∈ {1, 3} |

## 5. Milimo Integration

### 5.1 Architecture: Dedicated Microservice

SAM 3 runs as a separate FastAPI server, isolated from the main Milimo backend:

```
┌──────────────────────────────────────────────┐
│ Milimo Backend (port 8000)                    │
│   └─ InpaintingManager                        │
│       └─ HTTP POST → localhost:8001           │
├──────────────────────────────────────────────┤
│ SAM 3 Microservice (port 8001)                │
│   ├─ /health          → Status check          │
│   └─ /predict/mask    → Segmentation          │
└──────────────────────────────────────────────┘
```

**Rationale**: SAM 3 has different dependency trees and memory profiles from LTX-2/Flux. Running it in a separate process prevents model conflicts and allows independent scaling/restart.

### 5.2 Microservice Implementation (`sam3/start_sam_server.py`)

**Server**: FastAPI with `uvicorn`, bound to `0.0.0.0:8001`.

**Lifespan Startup**:
1.  Append current directory and parent to `sys.path` for import resolution.
2.  Import `build_sam3_image_model` from `sam3` package.
3.  **Device Selection**: `cuda` → `mps` → `cpu` (automatic detection).
4.  **Checkpoint Loading**: Loads from `<project_root>/backend/models/sam3.pt`.
5.  `enable_inst_interactivity=True` — activates the `inst_interactive_predictor` for session-based prediction.
6.  Model moved to device and set to `.eval()` mode.
7.  On shutdown: `sam_model = None` (GC handles cleanup).

**Endpoints**:

#### `GET /health`
```json
{"status": "running", "model_loaded": true}
```
Returns model availability. Useful for readiness probes.

#### `POST /predict/mask`
**Request** (multipart form):
| Field | Type | Description |
|---|---|---|
| `image` | `UploadFile` | Source image (PNG/JPEG) |
| `points` | `str` (JSON) | List of `[x, y]` coordinate pairs |
| `labels` | `str` (JSON) | List of labels (`1`=foreground, `0`=background). Default: `[1]` |
| `multimask` | `bool` | Whether to return 3 candidate masks. Default: `false` |

**Processing Flow**:
1.  Read uploaded image → PIL → numpy array (RGB).
2.  Parse `points` JSON → `np.array`, reshape to `(N, 2)` if needed.
3.  Parse `labels` JSON → `np.array`, handle scalar case.
4.  `predictor.set_image(np_image)` — encodes image features (cached for multiple predictions).
5.  `predictor.predict(point_coords, point_labels, multimask_output)` → `(masks, scores, logits)`.
6.  Extract best mask (`masks[0]`).
7.  Convert to binary PIL Image: `(mask * 255).astype(uint8)`.
8.  Return as `StreamingResponse` with `image/png` MIME type.

**Response**: Binary PNG image (single channel, 0/255 values).

**Error Handling**:
- 503 if model not loaded (checkpoint missing).
- 400 if points/labels JSON is malformed.
- 500 for prediction errors (with traceback in server log).

### 5.3 Client Integration (`backend/managers/inpainting_manager.py`)

**Class**: `InpaintingManager` (singleton `inpainting_manager`)

**Configuration**:
- `self.sam_url = f"http://localhost:{config.SAM_SERVICE_PORT}"` (default port `8001`).

#### `get_mask_from_sam(image_path, points)`
1.  Validates image exists on filesystem.
2.  Opens image as binary for multipart upload.
3.  Sends `POST {sam_url}/predict/mask` with:
    -   `files={"image": file_handle}`.
    -   `data={"points": str(points), "multimask": "false"}`.
4.  On 200 response:
    -   Saves mask PNG alongside source image: `{basename}_mask_{uuid6}.png`.
    -   Returns absolute path to saved mask.
5.  On failure: Logs error, returns `None`.

#### `process_inpaint(job_id, image_path, mask_path, prompt)`
1.  Opens source image as RGB, mask as grayscale (`"L"`).
2.  Calls `flux_inpainter.inpaint(image, mask, prompt, guidance=2.0, enable_ae=True, enable_true_cfg=False)`.
3.  **Output Location Logic**:
    -   Derives output dir from source image path.
    -   If source is in `assets/` → redirects output to `generated/`.
    -   Saves as `inpaint_{job_id}.jpg`.
4.  Returns output path.

### 5.4 End-to-End Inpainting Flow

```
User clicks on image in UI
        │
        ▼
Frontend sends click coordinates (points)
        │
        ▼
Backend: InpaintingManager.get_mask_from_sam()
        │ POST /predict/mask
        ▼
SAM 3 Microservice: Returns binary mask PNG
        │
        ▼
Backend: Saves mask to filesystem
        │
        ▼
Backend: InpaintingManager.process_inpaint()
        │ flux_inpainter.inpaint()
        ▼
Flux 2: RePaint-style inpainting with mask
        │
        ▼
Output: inpaint_{job_id}.jpg saved to project
        │
        ▼
SSE broadcast to frontend → UI refreshes
```

### 5.5 Weight Path & Configuration

| Config | Value | Description |
|---|---|---|
| Checkpoint | `<project_root>/backend/models/sam3.pt` | SAM 3 model weights |
| Port | `config.SAM_SERVICE_PORT` (default `8001`) | Microservice port |
| Device | Auto-detected: `cuda` > `mps` > `cpu` | Inference device |

### 5.6 CORS Configuration

The SAM 3 microservice allows **all origins** (`allow_origins=["*"]`). This is acceptable because:
1.  It only listens on `localhost`.
2.  It is never exposed to the internet in production.
3.  The main backend is the only client.

## 6. Debugging & Maintenance Guide

- **Model Not Loading**: Check that `backend/models/sam3.pt` exists. The server will start but `/predict/mask` will return 503 errors.
- **Connection Refused**: Ensure the SAM server is running on port 8001 before starting the main backend. Check `python sam3/start_sam_server.py`.
- **Mask Quality Issues**: Try adding background points (`label=0`) to exclude regions. SAM 3's single-point segmentation can be ambiguous.
- **Memory**: The SAM model stays resident. On shared-GPU systems, this may compete with Flux/LTX for VRAM. Consider `/health` monitoring.
- **Import Issues**: The server manipulates `sys.path` to find the `sam3` package. If the directory structure changes, update the path logic in `start_sam_server.py`.

## 7. Glossary

- **Zero-Shot Segmentation**: The ability to segment arbitrary objects without class-specific training. The model generalizes from its large-scale pretraining.
- **Instance Interactivity**: A session-based prediction mode where image features are computed once and cached, allowing multiple fast prediction calls with different prompts on the same image.
- **Promptable Segmentation**: A paradigm where the model takes spatial prompts (points, boxes) as input to determine what to segment, rather than relying on fixed semantic classes.
- **Multi-Mask Output**: Returning multiple candidate masks at different granularity levels (whole object, part, subpart), allowing the caller to choose the most appropriate level.
