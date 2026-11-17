---
type: concepts
created: 2026-10-09
updated: 2026-11-17
tags: memory-management vram slots ollama omlx strata unified-memory
sources:
  - docs/01_system_architecture.md
  - docs/03_ai_pipelines.md
  - docs/user_guide.md
  - ../../backend/memory_manager.py
  - ../../backend/services/model_loader.py
related:
  - ../models/ltx-2.md
  - ../models/flux-2.md
  - ../pipelines/chained-generation.md
  - ../systems/memory-management-audit-2026-11.md
---

# Memory Slots

## Definition

On Apple Silicon, GPU models share the same physical unified memory — there is no
separate VRAM to overflow into. The `MemoryManager` (`backend/memory_manager.py`)
enforces **mutual exclusion between heavy models** via named slots so two giants
never sit in memory at once.

## Where It Shows Up

| Slot | Model | Footprint |
|---|---|---|
| `video` | LTX-2 19B + Gemma 3 + VAE/upscalers | **~76 GB** |
| `image` | Flux 2 Klein 9B + Qwen3 encoder + AE/CLIP/IP | **~36 GB** |
| `llm_http` | Ollama / oMLX / Strata / LM Studio / vLLM | variable; measured 6.3 GB (Llama 3B) and **33 GB** (Strata Qwen3.8) |
| `sam` | SAM 3 microservice | 3.2 GB weights, ~6–8 GB runtime |
| `depth` | Depth Anything V2 Small | ~0.5 GB |

Rules: `video ↔ image` are mutually exclusive — `prepare_for("video")` unloads Flux
(and `llm_http`) before LTX-2 loads, and vice versa. Pipeline swaps follow: delete
reference → `gc.collect()` → empty GPU cache → `prepare_for(slot)` → load.

`llm_http` is best-effort: Ollama can be told to unload (`keep_alive: 0`), oMLX
exposes `/v1/models/{id}/unload`, and Strata currently exposes no unload we can
call. A warm model is still a resident model; the budget must include it even when
it cannot be released.

## Why It Matters

Without slot discipline, a prompt-enhancement call followed by video generation
would leave a 27B LLM and a 19B video model resident simultaneously. The slot
system is what makes the local-first studio feasible on consumer hardware.

**But slots are not a budget.** On 2026-11-17 this machine (128 GB, 115.4 GB MPS
ceiling) ran a 121-frame upscale job while Strata held 33 GB and oMLX 6.5 GB. The
sum exceeded the ceiling, swap hit 96%, macOS raised an out-of-memory dialog, and
the worker died. See
[[../systems/memory-management-audit-2026-11.md]] for the incident, model costs,
and the M0–M5 plan.

## Audit Notes (2026-11-17)

- All heavy jobs (video, image/thumbnail, inpaint, element, AI parse, depth) now go
  through `enqueue_gpu_job` → the global GPU semaphore. The older “direct dispatch”
  finding is resolved.
- The pre-check is still static: `VRAM_ESTIMATE_VIDEO_GB=40` and
  `IMAGE_GB=24` versus `psutil.available × 1.2`; real peaks are ~76 GB and ~36 GB.
- External engines are invisible to `MemoryManager`; only Ollama has an unload path
  wired into `runtime.release_local()`.
- `/models/{id}/activate` and `/deactivate` load/unload pipelines **outside** the job
  semaphore and can unload LTX during a running job.
- No idle TTL, no pressure/swap guard, no MPS cap, no peak sampler, and no
  per-model cost registry.
- A crashed worker left its Job/QueuedJob at `processing/running` and the reloader
  parent spinning; recovery only happens on restart.

## Related

- [[../systems/memory-management-audit-2026-11.md]] — measured budget, engine
  adapters, admission, eviction, and the implementation plan
- [[../systems/engineering-audit-2026-11.md]] — original slot enforcement findings
- [[../models/ltx-2.md]]
- [[../models/flux-2.md]]
- [[../pipelines/chained-generation.md]] — holds the video slot across chunks
