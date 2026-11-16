---
type: models
created: 2026-10-09
updated: 2026-10-09
tags: ltx-2 video-generation transformer
sources:
  - README.md
  - docs/03_ai_pipelines.md
  - docs/ltx2-bible.md
related:
  - ../pipelines/chained-generation.md
  - flux-2.md
  - ../systems/prompt-enhancement.md
  - ../systems/storyboard-engine.md
---

# LTX-2 (Video Generation)

## What It Does

LTX-2 19B is the dual-stream transformer behind all video generation: text-to-video,
image-to-video, and keyframe interpolation. Generation is two-stage: base generation
at half resolution with the distilled LoRA-384, then 2× spatial upsampling (optional
temporal upsampling doubles frames).

## How It's Wired In

- `ModelManager` (`backend/model_engine.py`) is the singleton loader; auto-selects
  checkpoint (full precision preferred, FP8 fallback on CUDA only).
- Three pipelines: `TI2VidTwoStagesPipeline` (default ti2vid), `ICLoraPipeline`
  (subject consistency via concept injection), `KeyframeInterpolationPipeline`.
- Pipeline selection in `tasks/video.py`: `num_frames == 1` bypasses LTX entirely and
  delegates to Flux 2; two images at different frames → keyframe; IC-LoRA elements →
  ic_lora; default → ti2vid; `num_frames > config.NATIVE_MAX_FRAMES` (**121** in
  current code) → chained task.
- Image conditioning replaces noisy latents at frame 0 via
  `image_conditionings_by_replacing_latent`, with priority: manual timeline image >
  concept-art thumbnail > previous shot's last frame (continuity).
- Text encoding via Gemma 3 (also the prompt-enhancement path).

## Checkpoints & Paths

| Artifact | Path |
|---|---|
| Full checkpoint | `LTX-2/models/checkpoints/ltx-2-19b-distilled.safetensors` |
| FP8 fallback (CUDA only) | `LTX-2/models/checkpoints/ltx-2-19b-distilled-fp8.safetensors` |
| Distilled LoRA-384 | `LTX-2/models/checkpoints/ltx-2-19b-distilled-lora-384.safetensors` |
| Spatial upscaler | `LTX-2/models/upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors` |
| Temporal upscaler | `LTX-2/models/upscalers/ltx-2-temporal-upscaler-x2-1.0.safetensors` |
| Gemma 3 text encoder | `LTX-2/models/text_encoders/gemma3/` |

## MPS/CUDA Notes

- FP8 is CUDA-only; MPS forces `fp8transformer=False`.
- VAE decode is cast to float32 on MPS — otherwise black frames.
- **Docs/README claim the transformer is forced to float32 on MPS, but current
  `model_engine.py` only casts the VAE** (`:145-148`). Either the forcing was lost
  in a refactor or the docs are stale; verify before trusting bf16 on MPS.
- `gc.collect()` + `torch.mps.empty_cache()` on every pipeline swap.

## Internals & Gotchas (from the LTX-2 bible)

- **Training & distillation**: trained with rectified flow matching, then distilled
  via rank-384 LoRA from ~50 steps to **~8 steps**. Stage 1 denoises 8 steps at low
  latent resolution (noise tensor shape `(B, C, T, H, W)`); Stage 2 does ×2 spatial
  upsampling, with an optional ×2 temporal upsampler doubling framerate.
- **Constructor** (`TI2VidTwoStagesPipeline`): `checkpoint_path`, `distilled_lora`,
  `spatial_upsampler_path`, `temporal_upsampler_path`, `gemma_root`, `loras`,
  `device`, `fp8transformer` — note `distilled_lora` and user `loras` are separate
  arguments.
- **IC-LoRA skips distillation**: it takes dedicated `loras` weight files and does
  *not* use the distilled LoRA — a different step regime, not the 8-step path.
- **Keyframe contract**: exactly 2 conditioning images (frame 0 → start, frame N →
  end).
- **Conditioning API**: `num_images_cond` is a list of `(path, frame_idx, strength)`
  tuples — the general mechanism is indexed by frame, not only frame 0; Milimo's
  frame-0 priority rule is the default-chunk convention on top of it.
- **Guidance**: typically `guidance_scale=1.0` for the distilled path.
- **Input URL resolution**: timeline images arrive as frontend URLs; the backend
  parses with `urlparse`, `unquote()`s (`%20` etc.), strips the `/projects` prefix,
  joins with `config.PROJECTS_DIR`, and validates with `os.path.exists()`.
- **Pipeline cache gotcha**: `ModelManager` caches by pipeline-type string — a
  mismatch like `"ti2vid"` vs `"TI2Vid"` silently reuses a stale pipeline.
- **MPS diagnostics**: black video → check `pipeline.vae.dtype == torch.float32`;
  bf16 transformer has historically thrown `MPSNDArrayMatrixMultiplication`, but the
  current loader does not force transformer float32 — check the actual pipeline
  dtype first. Fragmentation OOM → `gc.collect()` + `empty_cache()` on swap.
- **Repo layout**: `LTX-2/packages/ltx-core` (SafeTensor loading, LoRA merge,
  registry), `ltx-pipelines` (ti2vid_two_stages.py, ic_lora.py,
  keyframe_interpolation.py), `ltx-training` (unused at inference).
- **Output persistence**: video → `projects/{id}/generated/{job_id}.mp4`; thumbnail
  (first frame) → `projects/{id}/thumbnails/{job_id}.jpg`; Shot gets `video_url`,
  `thumbnail_url`, `actual_frames`, `status="completed"`.
- The bible omits the concept-art-thumbnail fallback in the conditioning priority
  chain — treat that chain as documented by the pipeline analysis, not the bible.

## Related

- [[../systems/engineering-audit-2026-11.md]] — current MPS/loader findings
- [[../pipelines/chained-generation.md]] — what happens past 121 frames (aka "Smart Continue")
- [[flux-2.md]] — the single-frame delegation target
- [[../systems/prompt-enhancement.md]] — `generate_enhanced_prompt()` and the `is_reference_only` contract
- [[../systems/storyboard-engine.md]] — `prepare_shot_generation()` continuity logic
