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
| `comfy` | ComfyUI external process | declared `MILIMO_COMFY_GB` (default 2 GB + 1 GB workspace); `/free` releases its cache |

Rules: `video ↔ image` are mutually exclusive — `prepare_for("video")` unloads Flux
(and `llm_http`) before LTX-2 loads, and vice versa. `comfy` conflicts with
`video`, `image`, and `llm_http`; an idle ComfyUI cache is released with `/free`.
Pipeline swaps follow: delete reference → `gc.collect()` → empty GPU cache →
`prepare_for(slot)` → load.

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

## Implemented behavior (2026-11-17)

Slots are now backed by a measured budget rather than name-only exclusion.

- **Budget** (`services/memory/budget.py`) reads total RAM, a 15% reserve, the MPS
  recommended working set, physical free, free swap, kernel pressure, in-process
  slots, external engines, and SAM. The cost registry estimates LTX-2 (76 GB base),
  Flux (36 GB base), SAM, depth, and LLM residency from parameters.
- **Admission** runs before every heavy load (video, image, chained, ai_parse,
  inpaint, element, activation, warm). It counts reclaimable idle slots and
  unloadable engines, applies a physical-free and swap-safety margin, and returns
  `retryable` plus a fitting `downgrade`.
- **Leases** protect running work: a slot with `leases > 0` is never evicted, and a
  conflicting slot in use refuses the load. Activation/deactivation is refused
  while a job is active.
- **Idle TTL** (video/image 300 s, LLM 120 s, SAM 600 s, depth 300 s, comfy 300 s)
  and pressure events evict idle slots; `keep-warm` pins a slot; leased slots are exempt.
- **Engine adapters** measure Ollama (`/api/ps`), oMLX (`/v1/models/status`), and
  use a declared `memory_gb` for Strata/generic engines, with unload where
  supported. Strata is still unloadable only by the user.
- **Peak sampler** records per-stage peak and swap for each job in `memory_json`.
- **MPS cap** (`set_per_process_memory_fraction`) makes PyTorch fail before macOS
  thrashes; encode runs off the event loop; load failures release the slot.
- **Scheduling** is priority-aware with same-slot grouping, a bounded “waiting for
  memory” retry, and an applied downgrade ladder before allocation.
- **SAM** loads lazily on the first segmentation request and can be released via
  `/unload`; its idle budget cost is 0.5 GB instead of 8 GB.

Remaining caveats: the cost registry is hand-tuned; Strata has no remote unload;
a cancelled job keeps its pipeline until the TTL; only the MPS decode path is
tiled. ComfyUI residency is declared, not measured per model, and uses port 8001 by
default; SAM 3 uses 8002. See [[../systems/memory-management-audit-2026-11.md]] for the incident,
measurements, and M0–M5 evidence.

## Related

- [[../systems/memory-management-audit-2026-11.md]] — measured budget, engine
  adapters, admission, eviction, and the implementation plan
- [[../systems/engineering-audit-2026-11.md]] — original slot enforcement findings
- [[../models/ltx-2.md]]
- [[../models/flux-2.md]]
- [[../pipelines/chained-generation.md]] — holds the video slot across chunks
