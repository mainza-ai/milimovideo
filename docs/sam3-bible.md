# SAM 3 Repository Documentation

## 1. Executive Summary

SAM 3 (Segment Anything Model 3) is Meta's state-of-the-art segmentation system, designed to provide **promptable, zero-shot segmentation** of arbitrary objects in images and video. SAM 3 supports **text prompts, point prompts, box prompts, and multi-object detection** — and extends to **video object tracking** with bidirectional propagation.

SAM 3 extends its predecessor (SAM 2) with:
- **Text-Prompted Segmentation**: Describe objects in natural language → masks, bounding boxes, and confidence scores.
- **Visual-Language Backbone**: Fused ViT + text encoder (`SAM3VLBackbone`) for joint vision-language understanding.
- **Video Object Tracking**: Track objects across video frames using `Sam3VideoPredictor` with temporal disambiguation.
- **Instance-Level Interactivity**: Built-in predictor for interactive point/click segmentation.
- **Multi-Object Detection**: Detect all instances of a concept with per-object confidence scoring.

Milimo integrates SAM 3 as a **standalone microservice** running on port 8001, supporting three distinct workflows: text-based detection, click-to-segment, and video tracking.

## 2. System Architecture Overview

### Architectural Style
SAM 3 uses an encoder-decoder architecture with a visual-language backbone:

1.  **Visual-Language Backbone** (`SAM3VLBackbone`): Fuses a Vision Transformer (ViT) for image features with a text encoder (VE Text Encoder + BPE tokenizer) for language understanding.
2.  **Sam3Processor**: High-level API for text-prompted and box-prompted segmentation. Handles `set_image()` → `set_text_prompt()` → masks/boxes/scores.
3.  **Instance Interactive Predictor**: Session-based point/click segmentation using cached image features.
4.  **Mask Decoder**: Lightweight transformer that outputs segmentation masks from encoded features + prompts.
5.  **Video Predictor** (`Sam3VideoPredictor`): Tracks objects across video frames with temporal disambiguation and memory-based propagation.

### Key Capabilities
| Capability | API | Description |
|---|---|---|
| **Text-Prompted Detection** | `Sam3Processor.set_text_prompt(state, "a dog")` | Returns masks + bounding boxes + confidence scores for all matching objects |
| **Box-Prompted Segmentation** | `Sam3Processor.add_geometric_prompt(box, label, state)` | Segment within a bounding box region |
| **Point-Based Segmentation** | `inst_interactive_predictor.predict(points, labels)` | Click-to-segment with foreground/background points |
| **Multi-Mask Output** | `predict(multimask_output=True)` | Returns multiple candidate masks ranked by confidence |
| **Video Object Tracking** | `Sam3VideoPredictor.propagate_in_video()` | Track objects across all frames bidirectionally |
| **Confidence Scoring** | `state["scores"]` | Per-mask confidence values for filtering/ranking |

## 3. Core Concepts

- **Sam3Processor**: The primary high-level API for text and box prompts. Handles image encoding, text forward pass, and grounding inference in a stateful pipeline.
- **Promptable Segmentation**: SAM takes spatial (points/boxes) or semantic (text) prompts and segments matching regions.
- **Foreground/Background Labels**: Points are labeled `1` (foreground = include) or `0` (background = exclude).
- **Multi-Mask vs Single-Mask**: When `multimask_output=True`, SAM returns multiple candidate masks. When `False`, returns the single best mask.
- **Confidence Threshold**: `Sam3Processor` applies a configurable threshold (default 0.5) to filter low-confidence detections.
- **Video Sessions**: The video predictor uses session-based tracking with `start_session` → `add_prompt` → `propagate_in_video` → `close_session`.

## 4. Model Architecture Details

| Component | Details |
|---|---|
| **Visual Backbone** | Vision Transformer (ViT) with FPN neck |
| **Text Encoder** | VE Text Encoder with BPE tokenizer (16M vocab) |
| **VL Backbone** | `SAM3VLBackbone` — fused vision + language features |
| **Segmentation Head** | Universal Segmentation Head with pixel decoder |
| **Geometry Encoder** | Encodes spatial prompts (points, boxes) into embedding space |
| **Dot Product Scoring** | MLP-based scoring module for mask ranking |
| **Tracker** | `Sam3TrackerPredictor` — memory-based video tracking with temporal disambiguation |
| **Checkpoint** | `backend/models/sam3/sam3.pt` (~3.4GB) |
| **Input** | RGB image (any resolution, internally resized to 1008px) |
| **Output** | Binary masks `[N, H, W]`, bounding boxes `[N, 4]`, scores `[N]` |

## 5. Milimo Integration

### 5.1 Architecture: Dedicated Microservice

SAM 3 runs as a separate FastAPI server, isolated from the main Milimo backend:

```
┌──────────────────────────────────────────────────────────┐
│ Milimo Backend (port 8000)                                │
│   ├─ InpaintingManager (text masks, point masks)          │
│   │   └─ HTTP POST → localhost:8001                       │
│   └─ TrackingManager (video tracking sessions)            │
│       └─ HTTP POST → localhost:8001                       │
├──────────────────────────────────────────────────────────┤
│ SAM 3 Microservice (port 8001)                            │
│   ├─ /health            → Status check                    │
│   ├─ /predict/mask      → Point/box segmentation          │
│   ├─ /detect            → Text-prompted multi-object      │
│   ├─ /segment/text      → Text → merged mask PNG          │
│   ├─ /track/start       → Start video tracking session    │
│   ├─ /track/prompt      → Add text/point/box prompt       │
│   ├─ /track/propagate   → Propagate across frames         │
│   └─ /track/stop        → Close session, free GPU         │
└──────────────────────────────────────────────────────────┘
```

**Rationale**: SAM 3 has different dependency trees and memory profiles from LTX-2/Flux. Running it in a separate process prevents model conflicts and allows independent scaling/restart.

### 5.2 Microservice Implementation (`sam3/start_sam_server.py`)

**Server**: FastAPI with `uvicorn`, bound to `0.0.0.0:8001`.

**Lifespan Startup**:
1.  Import `build_sam3_image_model` and `Sam3Processor` from `sam3` package.
2.  **Device Selection**: `cuda` → `mps` → `cpu` (automatic detection).
3.  **Checkpoint Loading**: Tries local `backend/models/sam3/sam3.pt` first, **auto-downloads from HuggingFace** if not found.
4.  `enable_inst_interactivity=True` — activates the `inst_interactive_predictor` for click-based segmentation.
5.  `Sam3Processor` initialized with model and default confidence threshold (0.5).
6.  Model set to `.eval()` mode.
7.  **Video predictor** is lazy-loaded on first `/track/*` request to conserve VRAM. The detected `device` is passed to `Sam3VideoPredictor(device=...)` so the model is placed on the correct device (CUDA, MPS, or CPU).

**Endpoints**:

#### `GET /health`
```json
{"status": "running", "model_loaded": true, "processor_ready": true}
```

#### `POST /predict/mask` — Point & Box Segmentation
| Field | Type | Description |
|---|---|---|
| `image` | `UploadFile` | Source image (PNG/JPEG) |
| `points` | `str` (JSON) | List of `[x, y]` coordinate pairs |
| `labels` | `str` (JSON) | `1`=foreground, `0`=background. Default: `[1]` |
| `multimask` | `bool` | Return multiple masks with scores. Default: `false` |
| `boxes` | `str` (JSON) | Optional: `[cx, cy, w, h]` normalized 0-1 |

**Response**: Binary PNG mask (single), or JSON with multiple base64 masks + scores when `multimask=true`.

#### `POST /detect` — Text-Prompted Multi-Object Detection
| Field | Type | Description |
|---|---|---|
| `image` | `UploadFile` | Source image |
| `text` | `str` | Natural language description ("a person", "red car") |
| `confidence` | `float` | Minimum confidence threshold (default 0.5) |

**Response**: JSON with array of detected objects, each containing `mask` (base64 PNG), `bbox` `[x0,y0,x1,y1]`, `score`, and `label`.

#### `POST /segment/text` — Text → Merged Mask PNG
Same params as `/detect`. Returns a single merged binary mask PNG — all matching objects ORed together. Designed for direct use with inpainting.

#### `POST /track/start` — Start Video Tracking Session
| Field | Type | Description |
|---|---|---|
| `video_path` | `str` | Absolute path to video file or frame directory |
| `session_id` | `str` | Optional custom session ID |

#### `POST /track/prompt` — Add Tracking Prompt
| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session from `/track/start` |
| `frame_idx` | `int` | Frame to place the prompt on |
| `text` | `str` | Text prompt (optional) |
| `points` / `boxes` | JSON | Spatial prompts (optional) |

#### `POST /track/propagate` — Propagate Tracking
| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Active session |
| `direction` | `str` | `"forward"`, `"backward"`, or `"both"` |
| `start_frame` | `int` | Starting frame index |
| `max_frames` | `int` | Max frames to track (-1 = all) |

**Response**: JSON with per-frame mask results and object counts.

#### `POST /track/stop` — Close Session
Releases GPU memory associated with the tracking session.

### 5.3 Client Integration

#### `InpaintingManager` (`backend/managers/inpainting_manager.py`)

| Method | Description |
|---|---|
| `get_mask_from_sam(image_path, points)` | Point-based: POST to `/predict/mask` → saves mask PNG |
| `get_mask_from_text(image_path, text, confidence)` | Text-based: POST to `/segment/text` → saves merged mask PNG |
| `detect_objects(image_path, text, confidence)` | Multi-object: POST to `/detect` → returns JSON with masks/boxes/scores |

#### `TrackingManager` (`backend/managers/tracking_manager.py`)

| Method | Description |
|---|---|
| `start_session(video_path, session_id)` | Start tracking session |
| `add_prompt(session_id, frame_idx, text, points, boxes, ...)` | Add prompt at frame |
| `propagate(session_id, direction, start_frame, max_frames)` | Propagate tracking |
| `stop_session(session_id)` | Close and free resources |

### 5.4 Backend Routes (`backend/routes/elements.py`)

All SAM-related edit routes are **multipart file-upload proxies** — they accept `image` (UploadFile) + form fields and forward directly to the SAM microservice. This allows the frontend to send captured video frames without requiring server-side file paths.

| Method | Route | Accepts | Description |
|---|---|---|---|
| POST | `/edit/segment` | `image` (file) + `points` + `labels` | Point-based segmentation preview (proxies to SAM `/predict/mask`) |
| POST | `/edit/detect` | `image` (file) + `text` + `confidence` | Text-based multi-object detection (proxies to SAM `/detect`) |
| POST | `/edit/segment-text` | `image` (file) + `text` + `confidence` | Text-based mask generation (proxies to SAM `/segment/text`) |
| POST | `/edit/inpaint` | JSON body (`InpaintRequest`) | Inpainting: supports `mask_path`, `text_mask` (SAM text), or `points` (SAM points). Creates a `Job` DB record for status polling via `/status/{job_id}`. |
| POST | `/track/start` | JSON body | Start video tracking session |
| POST | `/track/prompt` | JSON body | Add tracking prompt |
| POST | `/track/propagate` | JSON body | Propagate tracking |
| POST | `/track/stop` | JSON body | Stop tracking session |
| POST | `/edit/track/save` | JSON body (`TrackingSaveRequest`) | Export tracking results: saves base64 masks as PNG files to `exports/tracking/<session_id>/masks/` and writes `manifest.json`. |

### 5.5 Frontend Components

#### `MaskingCanvas.tsx` — 4-Mode Masking Tool

Accepts a `videoRef` prop from `CinematicPlayer` for cross-origin-safe frame capture. Uses `captureFrameBlob()` helper to draw the current video/image frame onto a temporary canvas and export as JPEG. All SAM calls are routed through the backend proxy (`/edit/detect`, `/edit/segment`) rather than directly to `:8001`. The text prompt input includes SAM3-specific guidance (e.g., "Describe object to mask with SAM3").

| Tool | Description |
|---|---|
| **Brush** | Manual mask painting (white = edit area) |
| **Eraser** | Erase mask regions |
| **Click** | Click on objects → `POST /edit/segment` → instant mask overlay |
| **Text** | Type description → `POST /edit/detect` → bounding boxes + masks with confidence scores |

Error feedback is displayed via toast notifications (auto-dismiss after 5s).

#### `TrackingPanel.tsx` — Video Object Tracking UI

Manages the full SAM3 video tracking workflow with a session-based lifecycle. Renders a slide-out control panel with mask overlay canvas.

| Phase | UI State | Description |
|---|---|---|
| **idle → starting** | Auto-start on mount | POST `/track/start` with `videoPath` |
| **prompting** | Text input or click mode | POST `/track/prompt` with text, points, or frame index |
| **propagating** | Progress indicator | POST `/track/propagate` with direction (forward/backward/both) |
| **done** | Frame navigation | Browse per-frame mask results with ◀/▶ controls |
| **error** | Error display | Retry or close |

- **Prompt modes**: Text prompts and point/click prompts (box prompts supported by backend but not yet exposed in UI)
- **Mask overlay**: `<canvas>` element with 35% opacity colored masks per tracked object
- **Mutually exclusive** with edit/inpainting mode in `CinematicPlayer`
- **Crosshair toggle** in `ControlsBar` to enter/exit tracking mode

#### `VideoSurface` (in `CinematicPlayer.tsx`)
- `<video>` and `<img>` elements include `crossOrigin="anonymous"` to prevent canvas tainting when frames are drawn for SAM processing.

#### `SegmentationOverlay.tsx` — Detection Visualization
- Renders colored mask overlays per detected object
- Bounding boxes with confidence score labels
- Click to select/deselect individual objects
- Integrates with `MaskingCanvas` in text-detection mode

### 5.6 End-to-End Flows

```
Text-Prompted Inpainting:
  User types "person" in masking tool
    → Frontend: POST /detect → SAM 3 Sam3Processor
    → Returns masks, bounding boxes, scores
    → User selects objects, clicks "Apply Mask"
    → Frontend: POST /edit/inpaint with text_mask
    → Backend: get_mask_from_text() → SAM /segment/text
    → Flux 2 RePaint inpainting

Click-to-Segment:
  User clicks on video frame
    → Frontend: POST /predict/mask with click coords
    → SAM 3 inst_interactive_predictor
    → Mask overlaid on canvas
    → User refines with additional clicks
    → Apply mask → inpainting workflow

Video Object Tracking:
  User selects video shot → /track/start
    → User clicks or types prompt → /track/prompt
    → "Track" button → /track/propagate
    → Per-frame masks returned
    → /track/stop to free resources
```

### 5.7 Weight Path & Configuration

| Config | Value | Description |
|---|---|---|
| Checkpoint | `backend/models/sam3/sam3.pt` (~3.4GB) | SAM 3 model weights |
| Safetensors | `backend/models/sam3/model.safetensors` (~3.4GB) | Alternative weight format |
| Port | `config.SAM_SERVICE_PORT` (default `8001`) | Microservice port |
| Device | Auto-detected: `cuda` > `mps` > `cpu` | Inference device |
| MPS Fallback | `PYTORCH_ENABLE_MPS_FALLBACK=1` (set in `run_sam.sh`) | CPU fallback for unsupported MPS ops |
| HF Auto-Download | `facebook/sam3` on HuggingFace | Fallback if local checkpoint missing |

### 5.8 MPS Compatibility (Apple Silicon)

SAM 3 was designed for CUDA. Running on MPS requires several workarounds:

#### Image/Detection Fixes
| Issue | Fix | Location |
|---|---|---|
| `grid_sample` crashes with empty tensors | Early return guard when `n_points == 0` | `geometry_encoders.py:_encode_points` |
| `pin_memory()` is CUDA-only | Replaced with direct `.to(device=...)` | `geometry_encoders.py:_encode_boxes` |
| `_assert_async` not implemented on MPS | `PYTORCH_ENABLE_MPS_FALLBACK=1` env var | `run_sam.sh` |
| FLASH/EFFICIENT SDPA backends unavailable | Restrict to `MATH` backend on MPS | `vl_combiner.py:_forward_text_no_ack_ckpt` |
| BFloat16 tensors incompatible with numpy | `.float()` cast before `.numpy()` | `start_sam_server.py:detect_objects` |
| Mask output shape `[N,1,H,W]` vs expected `[N,H,W]` | `.squeeze(1)` before numpy conversion | `start_sam_server.py:detect_objects` |
| Canvas tainting (cross-origin video frames) | `crossOrigin="anonymous"` on `<video>`/`<img>` | `CinematicPlayer.tsx:VideoSurface` |

#### Video Predictor Initialization Fixes
| Issue | Fix | Location |
|---|---|---|
| `Sam3VideoPredictor.__init__()` hardcodes `.cuda()` | Added `device` param; `.to(self.device)` with auto-detect | `sam3_video_predictor.py:__init__` |
| `_get_session_stats()` calls `torch.cuda.*` | Guarded behind `torch.cuda.is_available()` | `sam3_video_predictor.py` |
| `_get_torch_and_gpu_properties()` calls `torch.cuda.*` | Guarded behind `torch.cuda.is_available()` | `sam3_video_predictor.py` |
| `build_sam3_video_model` defaults to `"cpu"` | Changed to `"mps"` when MPS available | `model_builder.py:668` |
| bfloat16 params survive checkpoint loading | Convert all bfloat16 → float32 on MPS after `.to(device)` | `model_builder.py:797-812`, `sam3_video_predictor.py:69-76` |

#### Autocast & Dtype Fixes
| Issue | Fix | Location |
|---|---|---|
| CPU autocast silently falls back to bfloat16 | Disable autocast entirely on MPS via `contextlib.nullcontext()` | `sam3_tracking_predictor.py:50-67` |
| `torch.amp.autocast("cuda")` hardcoded | Dynamic device type; MPS → `"cpu"` with `enabled=False` | `decoder.py:70-72` |
| Explicit bfloat16 cast on backbone features | Conditional: MPS keeps float32, CUDA casts to bfloat16 | `sam3_image.py:834-836` |

#### CUDA-Hardcoding Fixes (Video Tracking)
| Issue | Fix | Location |
|---|---|---|
| `storage_device` hardcodes `torch.device("cuda")` | Uses `self.device` instead | `sam3_tracking_predictor.py:101` |
| `.cuda(non_blocking=True)` on mask logits | Replaced with `.to(device)` | `sam3_tracking_predictor.py:323` |
| `non_blocking=True` triggers CUDA init on MPS | Conditional: `non_blocking=(device.type == "cuda")` | `sam3_tracking_predictor.py` (6 sites) |
| `pin_memory()` in temporal position encoding | Skipped on non-CUDA devices | `sam3_tracker_base.py:167` |
| `pin_memory()` + `non_blocking` in output filtering | Conditional on CUDA | `sam3_video_inference.py:481` |

#### MPS Tensor Operation Fixes
| Issue | Fix | Location |
|---|---|---|
| `freqs_cis` on CPU vs query on MPS | Added `.to(xq_.device)` after reshape | `rope.py:70` |
| `repeat()` not supported for complex tensors | Decompose into real/imag, repeat each, reconstruct via `torch.complex()` | `rope.py:76-78` |
| `torch.stack` on empty tensor list | Guard for `batch_size == 0` with early return | `connected_components.py:43-48` |

#### Server Integration Fixes
| Issue | Fix | Location |
|---|---|---|
| `str(result)` destroys model output structure | Proper JSON serialization of `object_ids`, `scores`, `num_objects` | `start_sam_server.py:track_prompt` |
| Click points missing required `obj_id` | Auto-assign `obj_id=1` when points sent without explicit ID | `start_sam_server.py:track_prompt` |
| `max_frames=-1` sentinel causes 0 propagation iterations | Convert `-1` → `None` before passing to model | `start_sam_server.py:track_propagate` |
| `start_frame_idx` key mismatch | Fixed to `start_frame_index` matching predictor | `start_sam_server.py:track_propagate` |

#### Dependencies
| Package | Reason |
|---|---|
| `scikit-image` | Required for CPU-based connected components labeling (`skimage.measure.label`) used in hole-filling |

### 5.9 CORS Configuration

The SAM 3 microservice allows **all origins** (`allow_origins=["*"]`). This is acceptable because:
1.  It only listens on `localhost`.
2.  It is never exposed to the internet in production.
3.  The main backend and frontend are the only clients.

The frontend routes all SAM requests through the Milimo backend proxy (`:8000/edit/*`), which has explicit CORS for `localhost:5173`.

## 6. Debugging & Maintenance Guide

- **Model Not Loading**: Check that `backend/models/sam3/sam3.pt` exists. If missing, the server auto-downloads from HuggingFace (requires `huggingface_hub` installed and internet access).
- **Connection Refused**: Ensure the SAM server is running on port 8001 before starting the main backend. Start with `./run_sam.sh`.
- **Text Detection Returns Empty**: Lower the confidence threshold (default 0.5). Verify the text prompt matches visible objects in the image.
- **Mask Quality Issues**: Try adding background points (`label=0`) to exclude regions. Use the text tool for complex/ambiguous segmentation.
- **Video Tracking Memory**: The video predictor is lazy-loaded on first `/track/start` call. On shared-GPU systems, this may compete with the image model for VRAM.
- **Import Issues**: The server manipulates `sys.path` to find the `sam3` package. If the directory structure changes, update the path logic in `start_sam_server.py`.
- **MPS Crashes (Apple Silicon)**: Ensure `run_sam.sh` has `export PYTORCH_ENABLE_MPS_FALLBACK=1`. If new PyTorch ops fail, check for `pin_memory()`, empty tensor ops, complex tensor ops, or SDPA backend assumptions. See section 5.8 for the full compatibility table.
- **Video Tracking: 0 Detections**: If text prompt returns `Detected 0 object(s)`, verify the SAM server is returning proper JSON with `object_ids` (not `str(result)`). Check SAM logs for `Track prompt response: N objects detected`.
- **Video Tracking: 0 Propagation Iterations**: The model expects `None` for "track all frames", not `-1`. Ensure the server converts `-1` sentinels to `None` before calling the model.
- **Video Tracking: Device Mismatch**: Many SAM3 internals hardcode CUDA. Check for `.cuda()`, `non_blocking=True`, `pin_memory()`, and `torch.device("cuda")` — all must be conditionalized for MPS. See the CUDA-Hardcoding and MPS Tensor Op tables in section 5.8.
- **Inpaint Job Not Found (404)**: Inpaint jobs are now persisted as `Job` DB records (since Phase 5 fix). If `/status/{job_id}` returns 404, verify the `Job` record is created in `elements.py:inpaint_image()` and `update_job_db()` is called in `inpainting_manager.py`.

## 7. Glossary

- **Sam3Processor**: High-level API for text-prompted and box-prompted segmentation. Manages the full pipeline: image encoding → text forward → grounding inference.
- **Zero-Shot Segmentation**: Segment arbitrary objects without class-specific training.
- **Instance Interactivity**: Session-based prediction mode with cached image features for fast multi-prediction.
- **Visual-Language Backbone**: Fused ViT + text encoder architecture for joint vision-language understanding.
- **Temporal Disambiguation**: Video tracking heuristic that resolves identity conflicts when objects overlap or occlude each other across frames.
- **Bidirectional Propagation**: Tracking objects both forward and backward in time from the initial prompt frame.
- **Confidence Threshold**: Minimum score for a detection to be included in results. Default 0.5.
