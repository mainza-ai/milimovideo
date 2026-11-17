---
type: concepts
created: 2026-10-09
updated: 2026-10-09
tags: memory-management vram slots ollama
sources:
  - docs/01_system_architecture.md
  - docs/03_ai_pipelines.md
  - docs/user_guide.md
related:
  - ../models/ltx-2.md
  - ../models/flux-2.md
  - ../pipelines/chained-generation.md
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
| `video` | LTX-2 | ~76GB |
| `image` | Flux 2 | ~36GB |
| `llm` | Ollama (e.g. gemma3-27b) | variable, ~49GB for 27B |

Rules: `video ↔ image` are mutually exclusive — `prepare_for("video")` unloads Flux
before LTX-2 loads, and vice versa. Pipeline swaps always follow the sequence:
delete reference → `gc.collect()` → empty GPU cache → `prepare_for(slot)` → load.

The `llm` slot is managed differently: the Ollama provider sends its configured
`keep_alive` (default `0`, editable in Studio → AI & LLM) with every request, so the
model frees memory immediately instead of lingering on Ollama's default 5-minute
timeout. `keep_alive: 5m` is a real speed win (7.4 s → 2.2 s for a 3B on the
reference machine) but pins that model through denoising.

HTTP engines other than Ollama (oMLX, Strata, LM Studio, llama.cpp) are **outside**
`MemoryManager` today: their load/unload is owned by their own server. A warm 27B
MLX model plus LTX-2 19B does not fit on 32 GB. See
[[../systems/llm-local-first-next-2026-11.md]] (F10, L8-01).

## Why It Matters

Without slot discipline, a prompt-enhancement call followed by video generation would
leave a 27B LLM and a 19B video model resident simultaneously — an instant OOM on a
32GB machine. The slot system is what makes the whole local-first studio feasible on
consumer hardware. Frontend surfaces this honestly: switching from image work to video
work visibly unloads one model to load the other.

## Audit Notes (2026-11-16)

The slot design is right; enforcement is incomplete. Only video generation goes
through `gpu_job_wrapper` (`job_utils.py:41`). Thumbnails (`storyboard.py:426`),
still images (`jobs.py:268`), element visuals (`elements.py:92`), and inpainting
(`elements.py:161`) are dispatched directly, so Flux can run alongside LTX or two
Flux jobs can share the singleton. `prepare_for()` also marks a slot active without
loading anything, which makes `/models/{id}/activate` and the registry status lie.
See [[../systems/engineering-audit-2026-11.md]] (GEN-03, MOD-01) and
[[../roadmap/implementation-plan-2026-11.md]] (P2-09..P2-14).

## Related

- [[../systems/engineering-audit-2026-11.md]] — slot enforcement findings
- [[../models/ltx-2.md]]
- [[../models/flux-2.md]]
- [[../pipelines/chained-generation.md]] — holds the video slot across chunks
