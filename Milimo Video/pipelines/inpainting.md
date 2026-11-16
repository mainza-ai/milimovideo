---
type: pipelines
created: 2026-10-09
updated: 2026-10-09
tags: inpainting repaint sam-3 flux-2 masking
sources:
  - docs/01_system_architecture.md
  - docs/03_ai_pipelines.md
  - docs/05_execution_flow.md
  - docs/flux2-bible.md
  - docs/sam3-bible.md
related:
  - ../models/flux-2.md
  - ../models/sam-3.md
  - ../systems/sse-eventing.md
  - ../workflows/masking-tracking.md
---

# Inpainting Pipeline

## Trigger Conditions

`POST /edit/inpaint` with a JSON `InpaintRequest`: `image_path`, plus one of
`mask_path` (pre-made mask), `text_mask` (SAM text segmentation), or `points` (SAM
click segmentation), plus the `prompt` describing what goes in the masked area. The
request creates a `Job` DB record (type `inpaint`) so progress is pollable via
`/status/{job_id}`.

## Stages

1. **Mask acquisition** (if not supplied): `InpaintingManager` calls the SAM 3
   microservice — `get_mask_from_sam()` → `POST :8001/predict/mask` (click/points),
   `get_mask_from_text()` → `POST :8001/segment/text` (merged ORed mask), or
   `detect_objects()` → `POST :8001/detect` (masks + bboxes + scores for user
   selection). Masks saved as `{base}_mask_{uuid6}.png`.
2. **Dispatch**: `BackgroundTasks.add_task(process_inpaint, job_id, …)`; the job is
   registered in `active_jobs` for cancellation.
3. **RePaint loop** — `flux_inpainter.inpaint(image, mask, prompt, guidance=2.0,
   enable_ae=True, enable_true_cfg=False)`:
   - resize image + mask to multiples of 16; `ae.encode()` → original latents
     `x_orig`; mask flattened and downscaled to latent size `(H/16, W/16)` →
     `(1, L, 1)`; initialize `x` as pure noise;
   - each denoising step: predict the update, then blend
     `x = mask × x_pred + (1 − mask) × x_known`, where
     `x_known = t_prev × noise + (1 − t_prev) × x_orig` (the noisy-known
     formulation keeps unmasked regions faithful);
   - per-step `flux_callback` checks `active_jobs[job_id]["cancelled"]` and raises
     `RuntimeError("Cancelled")` to escape the GPU stack;
   - decode via `ae.decode()` (CPU-offloaded in float32 on MPS).
4. **Persist**: result saved to `projects/{id}/generated/inpaint_{job_id}.jpg`;
   `update_job_db()` marks the job completed; SSE `complete` fires with
   `{job_id, thumbnail_url, type: "inpaint", asset_id}`.

## Key Parameters

| Parameter | Value |
|---|---|
| `guidance` | 2.0 (internal guidance vector strength) |
| Resize constraint | image and mask multiples of 16 |
| Mask latent shape | `(H/16, W/16)` → reshaped `(1, L, 1)` |
| `enable_true_cfg` | False by default (double-pass costs 2×) |

## Failure Modes

- **Shape mismatch**: resolutions not divisible per the patch grid break the mask
  downscale — the multiple-of-16 resize is mandatory, not cosmetic.
- **MPS black output**: VAE decode must be CPU + float32; the inpaint path shares
  `denoise_inpaint` with generation, so all MPS guards apply.
- **Cancellation** is cooperative per denoising step (see [[../systems/sse-eventing.md]]).
- The flux2-bible documents no thumbnail/Asset/SSE detail for the inpaint path
  itself — the save path and SSE shape above come from the execution-flow and
  architecture docs. If behavior differs in code, code wins.

## Audit Notes (2026-11-16)

The stages above describe intent; the current code diverges. See
[[../systems/engineering-audit-2026-11.md]]:

- **No GPU queue.** `process_inpaint` is dispatched directly via `BackgroundTasks`,
  bypassing `gpu_job_wrapper`; it can run concurrently with LTX/Flux and OOM.
- **Event-loop blocking.** `process_inpaint` is async but calls the blocking
  `flux_inpainter.inpaint` inline; the backend cannot serve `/status` or cancel
  while inpainting.
- **No cancellation.** `step_callback` only reports progress; it never checks
  `active_jobs[job_id]["cancelled"]`. The request is not cancellable.
- **SAM calls block and have no timeout** (`requests.post` at
  `inpainting_manager.py:34/74/111`, called from async handlers).
- **Result is not attached to the shot.** The SSE `complete` carries
  `thumbnail_url` but no `url`; `ServerSlice` sets no `videoUrl`, and the inpainted
  image only appears in the gallery.
- **`guidance=2.0` is hardcoded**, ignoring the request.
- **`eval(req.points)`** in the route is an RCE (audit SEC-01).

## Related

- [[../systems/engineering-audit-2026-11.md]] — GEN-09/SEC-01 evidence
- [[../models/flux-2.md]] — the RePaint engine
- [[../models/sam-3.md]] — where masks come from
- [[../systems/sse-eventing.md]] — inpaint SSE payloads
- [[../workflows/masking-tracking.md]] — the user-facing flow
