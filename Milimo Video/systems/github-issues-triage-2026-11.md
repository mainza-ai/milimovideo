---
type: systems
created: 2026-11-17
updated: 2026-11-17
tags: github issues triage cuda vram flash-attention quantization wan api
sources:
  - ../../backend/services/memory/budget.py
  - ../../backend/services/memory/costs.py
  - ../../backend/models_manifest.json
  - ../../flux2/src/flux2/util.py
  - ../../backend/tasks/video.py
  - ../../backend/model_engine.py
related:
  - memory-management-audit-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
  - ../models/sam-3.md
---

# GitHub Issues Triage (2026-11-17)

The five open GitHub issues were filed in March 2026, before the memory budget,
typed event plane, Studio v2 shell, and model-management work. This page checks
each against `main` after the November implementation instead of assuming the
March behavior still holds.

| # | Issue | Current status | Evidence |
|---|---|---|---|
| 1 | Wan 2.2 + LoRAs | **Open — feature, not a regression** | no Wan adapter or weights; LTX/Flux pipeline paths are family-specific |
| 2 | Quantized models for low VRAM | **Open — feature** | no GGUF loader; `type: quantized` currently means LTX FP8 only |
| 3 | API for automation | **Resolved — documented** | all routes exist; new `docs/07_api.md` with batch/poll/SSE examples |
| 4 | Story Elements AE download 404 | **Fixed** | AE now comes from `FLUX.2-dev/ae.safetensors`; manifest repos corrected; clear error + diffusers fallback |
| 6 | RTX 5080 16 GB OOM on LTX-2 | **Mitigated — fails closed, no CPU offload** | CUDA VRAM is now the budget ceiling; 16 GB is refused with a VRAM reason before load |

## #4 — Story Elements, `ae.safetensors` link not working

**Root cause confirmed.** The vendored `flux2` loader asks for
`black-forest-labs/FLUX.2-klein-9B/resolve/main/ae.safetensors`, but the 9B repo
ships only `vae/` (diffusers) and no native `ae.safetensors`. The manifest also
pointed at `black-forest-labs/FLUX.2-klein`, a repo that does not exist. A user
without a local AE hit the 404 path and image generation failed.

**Fix:**

- `flux2/src/flux2/util.py`: every Flux 2 entry now carries
  `"ae_repo_id": "black-forest-labs/FLUX.2-dev"`, and `load_ae` downloads from
  that repo. The FLUX.2 AE does exist there; `FLUX.2-klein-9B` does not have the
  file.
- `backend/models_manifest.json`: the 9B checkpoint/components point at
  `black-forest-labs/FLUX.2-klein-9B`; the AE points at
  `black-forest-labs/FLUX.2-dev` and is marked gated (accept the license / set
  `HF_TOKEN`). The diffusers VAE in `FLUX.2-klein-9B/vae` remains the fallback.
- `load_ae` no longer calls `sys.exit(1)` on a download failure; it raises a
  clear `RuntimeError` naming the repo, the license/token action, and the
  diffusers fallback.

**Note on the reporter's suggestion:** `FLUX.1-dev/ae.safetensors` is a different
autoencoder from the FLUX.2 AE. It can produce incorrect colors/latents for
Flux 2, so the fix uses the FLUX.2 AE instead.

## #6 — RTX 5080 16 GB OOM

**What changed since March:** generation no longer loads a model before
admission. `memory_manager.admit_job` runs first and refuses or downgrades.

**Still a gap:** the budget was Apple-Silicon-shaped. On CUDA it used system RAM
as the ceiling and the 76 GB unified-memory LTX base, which is wrong in both
directions: a 16 GB card with 128 GB RAM could be admitted and OOM, while a
48 GB card with 32 GB RAM was refused unnecessarily.

**Fix (this pass):**

- `budget.snapshot` now reads `torch.cuda.mem_get_info()`: total VRAM, free VRAM,
  allocated VRAM, and a ceiling of `total × MILIMO_CUDA_MEMORY_FRACTION`
  (default 0.90, `MILIMO_CUDA_RESERVE_GB` floor). The snapshot reports
  `device: "cuda"` and a CUDA VRAM note.
- `costs.estimate` uses the CUDA bases: LTX FP8 **40 GB** and Flux 2 **18 GB**,
  instead of 76/36. The app auto-selects the FP8 LTX checkpoint on CUDA.
- Refusals say “VRAM” and include “CPU offload is not implemented; use a larger
  GPU or a smaller checkpoint.” A 16 GB card refuses a 121f job before any
  allocation; an 80 GB card admits it.
- `set_mps_cap` also sets `torch.cuda.set_per_process_memory_fraction`, so the
  process fails before the driver thrashes.
- `model_loader.can_load_model` uses free VRAM (`mem_get_info`) rather than
  total-minus-allocated and reports the card size.

**Not implemented:** `enable_model_cpu_offload` / `sequential_cpu_offload`. The
vendored LTX-2 pipelines expose no offload hooks (no `cpu_offload` symbol in
`LTX-2/packages`), and adding it means moving transformer/VAE components between
devices inside the pipeline, not a config flag. On a 16 GB card the honest
outcome is a refusal with a clear reason. A 24 GB card may still refuse the FP8
checkpoint; the next step would be a distilled/quantized checkpoint (see #2) or a
pipeline-level offload implementation.

**Recommendation for the reporter:** a 24 GB+ card is the practical CUDA
minimum for the FP8 19B checkpoint; 16 GB needs a quantized checkpoint (not yet
shipped) or CPU offload (not yet implemented). The refusal now happens before
the model loads, so it no longer leaves a wedged backend.

## #2 — Quantized models for low VRAM

**Status: open.** The registry has a `quantized` type and the FP8 LTX checkpoint,
but `FluxInpainter` loads a single full-precision `flux-2-klein-9b.safetensors`
through the vendored `flux2` package. There is no GGUF reader, no per-quant
pipeline, and no quantized manifest entries.

**Concrete plan:**

1. Add GGUF quantized Flux 2 entries to the manifest and registry (size, quality,
   recommended VRAM).
2. Load them through diffusers' GGUF path or a small `gguf` dequant loader, then
   map the state dict into `flux2.model.Flux2` (the custom package expects a
   full-precision state dict today).
3. Calibrate `costs.IMAGE_BASE_GB_CUDA` per checkpoint and add the quant to the
   take provenance.
4. Validate quality against the full checkpoint with the M5 benchmark
   (`backend/benchmark_models.py`) and a small fixture set.

This is a model-artifact and loader change, not a UI toggle; it needs the actual
GGUF files to test. See [[memory-management-audit-2026-11]] M5.

## #3 — API for automation

**Resolved.** Every Studio operation is a REST route, and the queue/SSE/memory
services are shared. Added `docs/07_api.md` with:

- project creation, video/image submit, preflight, status, SSE replay;
- a Python batch example (submit all, then poll);
- cancellation, queue pause/resume, and automation guardrails;
- auth (`MILIMO_API_TOKEN`, `?token=` for SSE), CORS, and `/docs`.

No server code was missing for this request; the gap was documentation.

## #1 — Wan 2.2 + LoRAs

**Status: open — a new model family, not a bug.** The video path is built around
LTX-2: `model_engine.py` constructs `TI2VidTwoStagesPipeline` /
`ICLoraPipeline` / `KeyframeInterpolationPipeline`; `tasks/video.py` assumes LTX
latent grids, Gemma 3 text encoding, `TilingConfig`, and LTX audio; chained
generation uses LTX's latent overlap math. Wan 2.2 uses a different DiT, VAE,
text encoder (umT5), and LoRA format.

Adding it means:

1. A `WanPipeline` adapter with its own loader and dtype/device guards.
2. A registry family `wan2.2` and checkpoint/LoRA entries (the manifest already
   models family/role/type).
3. A task branch that does not reuse LTX chunk/audio assumptions.
4. Memory costs (a 14B Wan model is much smaller than LTX-2 but still a heavy
   slot), preflight, and Setup UI.
5. Quality/time baselines.

None of the Wan weights are vendored, so this is a separate model-integration
lane, not a fix to the current issue.

## Verification

```
./milimov/bin/python -m pytest tests/test_memory_budget.py   # CUDA/AE tests
./milimov/bin/python -m ruff check backend tests
```

The CUDA tests pin: a 16 GB snapshot reports `device=cuda`, `total=16`,
`available=12`; the estimator uses the 40/18 GB CUDA bases; a 121f job on 16 GB
is refused with a “VRAM” reason and the offload suggestion; the same job is
admitted on an 80 GB card. The AE test pins the manifest and `FLUX2_MODEL_INFO`
to `FLUX.2-dev` / `FLUX.2-klein-9B`.

## Related

- [[memory-management-audit-2026-11.md]] — budget, M5, and the OOM incident
- [[../roadmap/implementation-plan-2026-11.md]] — open/deferred work
- [[../models/flux-2.md]] — Flux 2 loading and AE hot-swap
- [[../../docs/07_api.md]] — automation guide
