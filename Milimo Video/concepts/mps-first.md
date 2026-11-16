---
type: concepts
created: 2026-10-09
updated: 2026-10-09
tags: mps apple-silicon dtype vao optimization
sources:
  - README.md
  - docs/03_ai_pipelines.md
  - docs/user_guide.md
related:
  - memory-slots.md
  - ../models/ltx-2.md
  - ../models/flux-2.md
  - ../models/sam-3.md
---

# MPS-First Optimization

## Definition

Apple Silicon is treated as a first-class target, not a fallback. CUDA remains the
primary performance target, but every pipeline ships with the dtype casts, device
placements, and memory discipline MPS needs to produce correct output — because the
default failure mode on MPS isn't slowness, it's **black frames and NaNs**.

## Where It Shows Up

**LTX-2** (`model_engine.py`):
- FP8 unsupported on MPS → `fp8transformer=False`
- VAE decode cast to `torch.float32` → fixes black frames
- `bfloat16` weights cast to float32/float16 at pipeline construction
- `gc.collect()` + `torch.mps.empty_cache()` on every pipeline swap

**Flux 2** (`flux_wrapper.py`):
- VAE decode on CPU in float32 (black-image bug)
- Entire flow model forced to float32 (bf16 → NaNs)
- Reference images capped at 3 refs / 768² px on MPS (OOM guard)
- Cache flush before every denoise

**SAM 3** (`sam3/`, `run_sam.sh`):
- `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops with no MPS kernel (`_assert_async`)
- `Sam3VideoPredictor` patched to accept a `device` param instead of hardcoding
  `.cuda()`; all `torch.cuda.*` calls guarded behind `torch.cuda.is_available()`
- SDPA restricted to MATH backend on MPS; bfloat16 → float before `.numpy()`
- Device auto-detection order: CUDA → MPS → CPU

## Why It Matters

The studio's whole pitch is local-first filmmaking, and a large share of filmmakers
work on Macs. Shipping CUDA-only assumptions would silently corrupt output on the
machines the audience actually owns. The MPS fixes are boring, numerous, and
non-negotiable — they live in the engineering law for a reason (AGENTS.md §0–3).
Any new pipeline work must answer "what breaks on MPS?" before it merges.

## Related

- [[memory-slots.md]] — unified memory is why slots exist
- [[../models/ltx-2.md]], [[../models/flux-2.md]], [[../models/sam-3.md]]
