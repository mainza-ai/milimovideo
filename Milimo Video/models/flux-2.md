---
type: models
created: 2026-10-09
updated: 2026-10-09
tags: flux-2 image-generation ip-adapter inpainting
sources:
  - README.md
  - docs/03_ai_pipelines.md
  - docs/flux2-bible.md
related:
  - ltx-2.md
  - ../concepts/memory-slots.md
  - ../pipelines/inpainting.md
  - ../workflows/masking-tracking.md
---

# Flux 2 (Image Generation & Inpainting)

## What It Does

Flux 2 Klein 9B handles all still-image work: concept-art generation, element
visuals, single-frame video requests (LTX-2 delegates `num_frames == 1` here), and
RePaint inpainting for the masking/editing tools.

## How It's Wired In

- `FluxInpainter` singleton (`backend/models/flux_wrapper.py`) wraps the flow model,
  autoencoder, Qwen 3 8B text encoder, and IP-Adapter projector.
- **IP-Adapter** (CLIP ViT-L → 4-token projection via `ImageProjModel`) injects
  character/location identity into the latent space — this is what makes `@Hero`
  look like the same person across shots. `ElementManager` auto-detects trigger words
  in prompts and attaches the right reference images.
- **Reference conditioning (native AE)**: `get_reference_embeds()` encodes refs
  through the AE into temporal embedding space with offsets (`t_off = scale + scale × t`),
  concatenated to the denoising input.
- **AE hot-swap**: `enable_ae` flag toggles between the native AutoEncoder (supports
  reference conditioning) and the diffusers AE fallback; toggling triggers a full
  model unload + reload.
- **True CFG**: Flux 2 natively uses guided distillation (single pass, ignores
  negative prompts). Enabling True CFG runs a double-pass loop
  (`pred = pred_uncond + cfg_scale × (pred_cond − pred_uncond)`) at 2× inference cost;
  off by default (`cfg_scale=1.0`).
- **RePaint inpainting**: each denoising step blends `mask × predicted + (1 − mask) ×
  known`, where the known region is noised from the original latents — used by
  `InpaintingManager.process_inpaint()` after SAM 3 supplies the mask.
- Results saved as JPG (quality 95) + quarter-size thumbnails; an Asset DB record is
  created and an SSE `complete` event fires with the `asset_id`.

## Checkpoints & Paths

| Artifact | Path |
|---|---|
| Flow model (9B) | `backend/models/flux2/flux-2-klein-9b.safetensors` |
| Native AE (preferred) | `backend/models/flux2/ae.safetensors` |
| Diffusers AE fallback | `backend/models/flux2/vae/` |
| Text encoder (Qwen 3 8B) | `backend/models/flux2/text_encoder/` |
| IP-Adapter weights | `backend/models/flux2/ip-adapter.safetensors` |

## MPS/CUDA Notes

- VAE decode runs on CPU in float32 on MPS (black-image bug otherwise).
- Entire flow model forced to float32 on MPS (bf16 → NaNs).
- Reference images capped at 3 refs / 768² px on MPS to avoid OOM.
- `gc.collect()` + `torch.mps.empty_cache()` before every denoise.

## Internals & Gotchas (from the Flux 2 bible)

- **Init**: the `flux_inpainter` singleton auto-initializes on import, tracking
  `model_loaded`, `using_native_ae`, `ip_adapter_loaded`, and
  `last_ae_enable_request` (the AE-toggle memo). `load_model(enable_ae)` sets
  weight-path env vars dynamically (`KLEIN_9B_MODEL_PATH`, `AE_MODEL_PATH`,
  `QWEN3_8B_PATH`, `QWEN3_8B_TOKENIZER_PATH`); `self.dtype = torch.float32` is
  forced in `__init__` on MPS (whole flow model, not per-call).
- **Denoise loop**: `denoise_inpaint()` *replaces* the standard `sampling.denoise()`
  for both generation and inpainting. `generate_image()` defaults:
  `guidance=2.0`, `num_inference_steps=25`, random int64 seed, `enable_ae=True`,
  `enable_true_cfg=False`. Reference tokens concatenate into the model input:
  `model_input = cat(x, img_cond_seq)`. A `flux_callback` checks
  `active_jobs[job_id]["cancelled"]` at every processing layer.
- **IP-Adapter detail**: one 1024-dim CLIP embedding → `Linear(1024 → 4×4096)` →
  reshape → `LayerNorm(4096)` = 4 Flux hidden-state tokens per reference image,
  concatenated into the input sequence. In True-CFG mode the IP tokens are
  **stripped in the uncond pass** and included only in the cond pass.
- **AE hot-swap detail**: `FluxAEWrapper.decode()` forces CPU + float32 for the
  decode step only. The native AE's `BatchNorm2d` normalizes latents to
  zero-mean/unit-variance before the flow model. Any `enable_ae` toggle change →
  full unload + reload.
- **Inpaint call**: `inpaint(image, mask, prompt, guidance=2.0, enable_ae=True,
  enable_true_cfg=False)` — resize to multiples of 16, `ae.encode()` → `x_orig`,
  mask downscaled to `(H/16, W/16)` → `(1, L, 1)`, start from pure noise, RePaint
  blend per step.
- **Config keys**: `FLUX_WEIGHTS_PATH = <PROJECT_ROOT>/backend/models/flux2`,
  `FLUX_IP_ADAPTER_PATH`. Pixel caps: 768² on MPS; CUDA 2024² single-ref / 1024²
  multi-ref.
- **Variant ambiguity (flagged)**: the bible lists two 9B variants (distilled Klein
  9B at 4 default steps vs Base 9B at 50) without saying which
  `flux-2-klein-9b.safetensors` is; Milimo's 25-step default matches neither
  upstream default. Not a contradiction — just unresolved.
- **Not enabled**: the `imwatermark` watermarking module is commented out; quantized
  weights aren't natively supported.

## Related

- [[ltx-2.md]] — delegates single frames here
- [[../concepts/memory-slots.md]] — Flux and LTX-2 are mutually exclusive slots
- [[../pipelines/inpainting.md]] — the RePaint pipeline
- [[../workflows/masking-tracking.md]] — masking and tracking flows
