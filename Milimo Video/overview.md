---
type: overview
created: 2026-10-09
updated: 2026-10-09
tags: overview local-first nle open-source
sources:
  - README.md
related:
  - systems/architecture.md
  - workflows/getting-started.md
---

# Overview: Milimo Video

**Milimo Video** is an open-source, AI-native cinematic studio — a professional
**non-linear editor (NLE)** for filmmakers, running entirely **local-first** on the
filmmaker's own machine. Apache 2.0 licensed. Author: Mainza Kangombe.

Unlike simple prompt-to-video interfaces, Milimo unifies best-in-class foundation
models into one cohesive filmmaking workflow:

| Model | Role |
|---|---|
| **LTX-2 19B** | Dual-stream transformer for cinematic video generation (text-to-video, image-to-video, keyframe interpolation) |
| **Flux 2 Klein 9B** | High-fidelity image synthesis with IP-Adapter reference conditioning and RePaint inpainting |
| **SAM 3** | Text-prompted segmentation, click-to-segment, and video object tracking (standalone microservice) |
| **Gemma 3** | Intelligent prompt enhancement and narrative direction |

## What makes it a studio, not a demo

- **Visual conditioning & character consistency** — Story Elements (Characters,
  Locations, Objects) with trigger words (e.g. `@Hero`); the system auto-detects
  triggers in prompts and injects IP-Adapter reference images, so identities persist
  across shots.
- **Storyboard engine** — paste screenplay text; Milimo parses it into scenes and
  shots (regex `ScriptParser` or Gemma 3 AI parsing), matches `@Elements`, and can
generate Flux 2 concept-art thumbnails as pre-viz.
- **Chained generation (Quantum Alignment)** — shots past the 121-frame native
  chunk generate autoregressively with a 25-frame overlap; the current code passes a
  pixel `media_sequence` and trims the overlap, not the older latent-tail handoff
  described in early docs.
- **Professional NLE** — magnetic 3-track timeline (V1 magnetic, V2 overlay, A1
audio), frame-accurate trimming, snap engine, undo/redo.
- **In-painting & intelligent editing** — mask anything via click or natural language
  (SAM 3), then add/remove elements with Flux 2 RePaint inpainting.
- **Memory discipline** — the MemoryManager enforces one heavy GPU model at a time;
  Ollama unloads immediately after prompt enhancement. Built MPS-first for Apple
  Silicon, CUDA as primary target.

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 19.2, TypeScript, Vite, Zustand (7-slice store + persist + zundo) |
| Backend | FastAPI (Python 3.10+), SQLModel/SQLAlchemy (SQLite), SSE via sse-starlette |
| Video AI | LTX-2 19B — 3 pipelines + chained generation |
| Image AI | Flux 2 Klein 9B — IP-Adapter, True CFG, RePaint |
| Segmentation | SAM 3 microservice (port 8001) |
| Processing | FFmpeg — thumbnails, frame extraction, overlap trimming, concat |

## Current State (2026-11-16)

The Nov 2026 engineering audit found the studio layer is not yet trustworthy:
timeline order and conditioning do not round-trip, export timing/compositing/audio
are wrong, and the local API is open. The model layer and workflow concepts are
strong. Read [[systems/engineering-audit-2026-11.md]] before trusting the NLE with a
real project; the fix sequence is
[[roadmap/implementation-plan-2026-11.md]].

## Related

- [[systems/engineering-audit-2026-11.md]] — current-state audit
- [[roadmap/implementation-plan-2026-11.md]] — prioritized fix plan
- [[systems/architecture.md]] — how the pieces fit together
- [[workflows/getting-started.md]] — from zero to studio
