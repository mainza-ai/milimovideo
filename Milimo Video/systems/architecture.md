---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: architecture frontend backend microservice sse
sources:
  - docs/01_system_architecture.md
  - docs/06_file_dependency.md
  - docs/CODEBASE_ANALYSIS.md
related:
  - ../overview.md
  - ../models/sam-3.md
  - ../concepts/memory-slots.md
  - ../decisions/sam3-microservice.md
  - data-models.md
  - frontend-state.md
  - sse-eventing.md
---

# System Architecture

## Purpose

The split architecture that keeps UI responsive while GPU-heavy inference runs
locally: a React frontend, a FastAPI backend on port 8000, and SAM 3 isolated as its
own microservice on port 8001.

## Components

- **Frontend (React 18 / Vite / TypeScript, port 5173)** — UI, timeline state,
  multi-track NLE, preview player. Key pieces: `CinematicPlayer`, `VisualTimeline`,
  `InspectorPanel`, `MediaLibrary`, `ElementsView`, `StoryboardView`, `SSEProvider`.
- **Backend (Python / FastAPI, port 8000)** — model inference, asset management, job
  queue, SQLite via SQLModel. **Seven routers / 93 routes** (verified 2026-11-16):
  projects, jobs, assets, elements, storyboard, shots, models.
- **Task engine** — `tasks/video.py`, `tasks/chained.py`, `tasks/image.py`;
  `ModelManager` singleton loads LTX-2 pipelines; `FluxInpainter` singleton wraps
  Flux 2; `StoryboardManager` coordinates chunks; `ElementManager`/`ElementMatcher`/
  `InpaintingManager` handle elements and masks.
- **SAM 3 microservice (port 8001)** — own FastAPI server; endpoints for health,
  mask prediction, detection, text segmentation, and video tracking sessions.
- **MemoryManager** — central mutual-exclusion between GPU models (see
  [[../concepts/memory-slots.md]]).

## Key Interactions

| Interaction | Protocol | Notes |
|---|---|---|
| UI → Backend | HTTP REST | CRUD for projects/shots/elements/assets; job dispatch |
| Backend → UI | SSE | Real-time progress, completion, errors; auto-reconnect with backoff |
| UI refresh sync | `GET /status/{job_id}` | One-shot job recovery on page load via `jobPoller` |
| Backend → SAM 3 | HTTP REST (port 8001) | Segmentation and tracking calls |
| Worker → LTX-2 / Flux 2 | Direct Python import | One pipeline cached at a time; MemoryManager gates swaps |
| Prompt enhancement | `llm.py` dispatcher | Gemma built-in or Ollama HTTP API (`keep_alive: 0` unloads after use) |
| Media serving | HTTP Range requests | `206 Partial Content` for Safari-compatible mp4/mp3 |

## Gotchas

- Only one LTX-2 pipeline is cached at a time; switching types triggers unload →
  `gc.collect()` → cache flush → reload.
- `num_frames == 1` silently delegates video generation to Flux 2 image generation.
- Ollama sends `keep_alive: 0` so the LLM unloads immediately after prompt
enhancement — otherwise a 27B model can sit on ~49GB VRAM.
- Static media must be served with range requests or Safari playback breaks.

## Module Map (from the dependency analysis)

**Backend** (`backend/`): `server.py` (FastAPI entry; owns `GET /events` SSE
directly), `config.py` (pure config, zero deps), `database.py` (ORM + sessions),
`events.py` (SSE broadcast), `job_utils.py` (job lifecycle: database + events +
file_utils), `schemas.py` (all Pydantic), `model_engine.py` (ModelManager),
`memory_manager.py`, `llm.py`, `models/flux_wrapper.py`, `managers/`
(element/inpainting/tracking), `services/` (script_parser, ai_storyboard,
element_matcher), `storyboard/manager.py`, `tasks/` (video, chained, image),
`routes/` — **7 routers**: projects, jobs, assets, elements, storyboard, shots,
models (the generate/edit routes live under jobs/elements/models). `routes/shots.py`
currently registers a duplicate `PATCH /shots/{shot_id}` that is shadowed by
`routes/projects.py`; see the audit.

**Frontend** (`web-app/src/`): `main.tsx` → `App.tsx` → Layout, CinematicPlayer,
SSEProvider. `stores/timelineStore.ts` (God Store, 20+ importers) composes 7 slices
in `stores/slices/`; `stores/types.ts` mirrors the backend models. Key utils:
`timelineUtils.ts` (magnetic layout), `snapEngine.ts`, `jobPoller.ts`,
`GlobalAudioManager.ts`, `config.ts` (API_BASE_URL).

**Key import edges**: `tasks/video.py` is the generation funnel (max fan-out: 8
imports); `tasks/image.py` → `flux_wrapper.py` → `memory_manager.py` (slot
discipline); `routes/storyboard.py` → `ai_storyboard.py` → `model_engine.py`
(Gemma via LTX encoder); `routes/elements.py` → inpainting/tracking managers →
SAM 3 over HTTP; `storyboard/manager.py` → `element_manager.py` →
`flux_wrapper.py` (trigger injection → IP-Adapter).

**Notes**: DB file is `backend/milimovideo.db`; the dependency doc counts 24 backend
files while the codebase analysis counts 28 — at least one doc predates the other
(`llm.py`/`memory_manager.py` era). The dependency graph also claims "no circular
dependencies" while showing `model_engine ⇄ memory_manager` edges — the slot
handshake is genuinely bidirectional by design.

## Audit Notes (2026-11-16)

The code-level audit found the architecture's main risk is not the model layer but
the studio contract: V1 order and conditioning do not round-trip, export timing is
derived from `start_frame` (all zero in the live DB), and image/inpaint jobs bypass
the GPU semaphore. Security is also open (CORS `*`, no auth, `eval` in inpaint).
See [[engineering-audit-2026-11.md]] for evidence and
[[../roadmap/implementation-plan-2026-11.md]] for the fix sequence.

## Related

- [[engineering-audit-2026-11.md]] — current-state findings
- [[../overview.md]]
- [[../models/sam-3.md]]
- [[../concepts/memory-slots.md]]
- [[../decisions/sam3-microservice.md]]
- [[data-models.md]] — the persistence layer in detail
- [[frontend-state.md]] — the store in detail
- [[sse-eventing.md]] — the event system in detail
