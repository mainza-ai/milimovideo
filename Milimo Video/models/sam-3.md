---
type: models
created: 2026-10-09
updated: 2026-11-17
tags: sam-3 segmentation tracking microservice
sources:
  - README.md
  - docs/01_system_architecture.md
  - docs/03_ai_pipelines.md
  - docs/sam3-bible.md
related:
  - ../systems/architecture.md
  - ../decisions/sam3-microservice.md
  - ../workflows/masking-tracking.md
  - ../pipelines/inpainting.md
---

# SAM 3 (Segmentation & Tracking)

## What It Does

Text-prompted segmentation, click-to-segment, and bidirectional video object tracking
— the "intelligent scissors" behind masking, inpainting, and selective editing.

## How It's Wired In

- Runs as an **isolated FastAPI microservice** on port 8002
  (`sam3/start_sam_server.py`); the backend talks to it over HTTP. See
  [[../decisions/sam3-microservice.md]] for why.
- Three predictors: `Sam3Processor` (text/box), `inst_interactive_predictor`
  (click/point), `Sam3VideoPredictor` (tracking; lazy-loaded on first `/track/*`
  call, device auto-detected CUDA → MPS → CPU).
- Backend clients: `InpaintingManager` (masks, text masks, detection) and
  `TrackingManager` (session-based tracking).
- Tracking UI (`TrackingPanel.tsx`) lifecycle: start → prompt → propagate → navigate
  results → save/load; mutually exclusive with edit mode.
- Checkpoint `backend/models/sam3/sam3.pt` (~3.4GB) auto-downloads from HuggingFace
  on the first segmentation request if missing. The image model is **lazy**: the
  service starts at ~0 GB and loads under a lock on demand (`POST /load` warms it,
  `POST /unload` releases it). The memory budget counts 0.5 GB idle, 8 GB loaded.

## API

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | — | `{status, model_loaded, processor_ready, loading, lazy, error}` |
| `POST /load` | — | warm the image model (idempotent) |
| `POST /unload` | — | release the image model + flush caches |
| `POST /predict/mask` | image + points/labels (+boxes) | binary PNG mask |
| `POST /detect` | image + text + confidence | `{objects: [{mask, bbox, score, label}]}` |
| `POST /segment/text` | image + text + confidence | merged binary PNG mask (for inpainting) |
| `POST /track/start` | video_path | `{session_id, status}` |
| `POST /track/prompt` | session_id + frame_idx + text/points/boxes | `{status: "prompt_added"}` |
| `POST /track/propagate` | session_id + direction + start_frame + max_frames | `{frames: [{frame_idx, masks}]}` |
| `POST /track/stop` | session_id | `{status: "closed"}` |

## MPS/CUDA Notes

- `PYTORCH_ENABLE_MPS_FALLBACK=1` set in `run_sam.sh` for ops not yet on MPS
  (`_assert_async`).
- `Sam3VideoPredictor` was patched to take a `device` param instead of hardcoding
  `.cuda()`; `torch.cuda.*` calls guarded behind `torch.cuda.is_available()`.
- SDPA restricted to the MATH backend on MPS; bfloat16 outputs cast to float before
  `.numpy()`.

## Internals & Gotchas (from the SAM 3 bible)

- **Startup order** (`start_sam_server.py`, bound to `0.0.0.0:8001`): the lifespan
  starts immediately with no model. The first image request calls `_ensure_model()`
  → import `build_sam3_image_model` + `Sam3Processor` → device auto-detect (CUDA →
  MPS → CPU) → load `sam3.pt` (auto-download from `facebook/sam3` if missing) →
  `enable_inst_interactivity=True` for click prediction → `Sam3Processor` (default
  confidence 0.5) → `.eval()`. `/load` can warm it; `/unload` releases it. The
  video predictor lazy-loads on the first `/track/*` call. Start `./run_sam.sh`
  **before** the backend when using segmentation.
- **Predictor internals**: `Sam3Processor` is stateful — `set_image()` →
  `set_text_prompt()` → masks/boxes/scores. Point labels: `1` = foreground,
  `0` = background. `multimask_output=True` returns ranked candidates; input is
  resized internally to 1008px. Architecture: ViT backbone (FPN neck) + VE text
  encoder with BPE tokenizer (16M vocab), pixel-decoder segmentation head,
  geometry encoder for spatial prompts. The video tracker is memory-based with
  **temporal disambiguation** for overlapping/occluding objects.
- **Tracking mechanics**: `/track/start` accepts a video file *or a frame
  directory*; `/track/propagate` takes `direction` = forward/backward/**both**
  (bidirectional supported). `max_frames=-1` is a sentinel the server converts to
  `None` ("all frames"). Click prompts without an object ID auto-assign `obj_id=1`.
  `/track/stop` releases the session's GPU memory.
- **Proxy routes** (`backend/routes/elements.py`) — the frontend never calls :8001
  directly: `POST /edit/segment` → `/predict/mask`; `POST /edit/detect` →
  `/detect`; `POST /edit/segment-text` → `/segment/text` (merged ORed mask);
  `POST /edit/inpaint` takes a **JSON body** (`InpaintRequest`), not multipart;
  `POST /track/*` take JSON bodies. CORS on :8001 is `allow_origins=["*"]`,
  acceptable because it only listens on localhost.
- **Save/load**: `POST /edit/track/save` exports base64 masks as PNGs to
  `exports/tracking/<session_id>/masks/` plus a `manifest.json`.
- **MPS gotcha catalog** (the full port): `non_blocking=True` conditionalized to
  CUDA at 6 sites (it triggered CUDA init on MPS); autocast disabled on MPS via
  `nullcontext()`; bfloat16 backbone casts conditional (MPS keeps float32);
  `freqs_cis` device mismatch fixed in `rope.py`; `torch.stack` on empty list
  guarded; `build_sam3_video_model` defaults to `"mps"` when available.
- **Doc gap (flagged)**: `01_system_architecture.md` lists a `POST /track/remove`
  endpoint, but the SAM 3 bible documents only
  start/prompt/propagate/stop — the bible may simply be incomplete. Verify against
  `sam3/start_sam_server.py` before relying on either.

## Related

- [[../systems/architecture.md]]
- [[../decisions/sam3-microservice.md]] — why it's a separate service
- [[../workflows/masking-tracking.md]] — the user-facing flows
- [[../pipelines/inpainting.md]] — inpainting via SAM masks
