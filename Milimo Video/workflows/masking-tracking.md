---
type: workflows
created: 2026-10-09
updated: 2026-11-17
tags: masking tracking inpainting sam-3 segmentation
sources:
  - docs/01_system_architecture.md
  - docs/sam3-bible.md
  - docs/user_guide.md
related:
  - ../models/sam-3.md
  - ../pipelines/inpainting.md
  - ../systems/sse-eventing.md
---

# Masking & Tracking

## Goal

Selectively edit video: mask any object (by brush, click, or natural language) and
inpaint it away or replace it — or track an object across every frame for precise
per-frame work.

## Prerequisites

- SAM 3 microservice running (`./run_sam.sh`, port 8002) — start it before the
  backend, or calls fail with "Connection Refused".
- A frame selected in Studio **Edit** (or the legacy editor). The frontend captures
  it locally; the Studio buttons are **Edit frame**, **Track**, and **Depth**.

## Steps — Masking & Inpainting

1. **Enter edit mode** — in Studio Edit, click **Edit frame** (the legacy editor
   uses the pencil/MaskingCanvas path). Tools: **Brush** (paint the mask, white =
   edit area), **Eraser**, **Click** (click an object), **Text** (describe it,
   e.g. "the sky").
2. **Generate the mask** — all SAM calls go through the backend proxy, never
   directly to :8001:
   - Click → `POST /edit/segment` → SAM `/predict/mask` (`inst_interactive_predictor`); refine with foreground/background clicks (labels 1/0)
   - Text → `POST /edit/detect` → SAM `/detect` (`Sam3Processor`); pick objects in `SegmentationOverlay` (colored overlays, bboxes + confidence scores), then "Apply Mask"
   - The merged-mask shortcut: `POST /edit/segment-text` → SAM `/segment/text` returns one ORed mask, built for direct inpainting
   - Toasts (auto-dismiss 5s) report SAM errors
3. **Inpaint** — type the replacement prompt ("a red sports car") →
   `POST /edit/inpaint` (JSON: `image_path` + `mask_path`/`text_mask`/`points` +
   `prompt`) → Flux 2 RePaint loop → `projects/{id}/generated/inpaint_{job_id}.jpg`.
   The inpaint is a persisted `Job`; poll `/status/{job_id}` or watch SSE `complete`
   (`type: "inpaint"`).

## Steps — Object Tracking

1. **Enter tracking mode** — **Track** in Studio Edit (legacy: crosshair toggle in
   `ControlsBar`). Tracking and edit mode are mutually exclusive.
2. **Start** — `TrackingPanel` auto-fires `POST /track/start` with the video path on
   mount (accepts a video file or a frame directory).
3. **Prompt** — text or click on frame N → `POST /track/prompt` (`session_id`,
   `frame_idx`, text/points/boxes). Box prompts exist in the backend but aren't
   exposed in the UI yet.
4. **Propagate** — `POST /track/propagate` with direction `forward`/`backward`/`both`
   and `max_frames` (`-1` = all frames; the server converts the sentinel to `None`).
   Browse per-frame results with ◀/▶; overlays render at 35% opacity per object.
5. **Save** — `POST /edit/track/save` exports PNG masks to
   `exports/tracking/<session_id>/masks/` plus a `manifest.json`; reload later via
   `/edit/track/load`. `POST /track/stop` closes the session and frees GPU memory.
   Studio also supports a local **Depth** overlay from Depth Anything V2 Small.

## Tips

- Lower SAM's confidence threshold (default 0.5) when text detection comes back empty.
- Add background points (label 0) to sharpen click masks.
- Tracking and the image model can fight over VRAM on shared-GPU machines — run
  tracking with nothing else generating.
- Box prompts are backend-ready; if you need them in the UI, that's a
  `TrackingPanel` change, not a SAM change.

## Related

- [[../models/sam-3.md]] — the segmentation engine
- [[../pipelines/inpainting.md]] — the RePaint mechanics
- [[../systems/sse-eventing.md]] — inpaint job events
