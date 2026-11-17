---
type: systems
created: 2026-10-09
updated: 2026-11-17
tags: architecture frontend backend microservice sse alembic studio
sources:
  - ../docs/01_system_architecture.md
  - ../docs/06_file_dependency.md
  - ../web-app/src/
  - ../backend/
related:
  - ../overview.md
  - ../models/sam-3.md
  - ../concepts/memory-slots.md
  - ../decisions/sam3-microservice.md
  - data-models.md
  - studio-shell.md
  - frontend-state.md
  - sse-eventing.md
---

# System Architecture

## Purpose

The split architecture that keeps the UI responsive while GPU-heavy inference runs
locally: a React Studio shell, a FastAPI backend on port 8000, an optional SAM 3
microservice on port 8001, and a Tauri desktop wrapper that hosts the same shell.

## Components

- **Studio frontend (React 19.2 / Vite / TypeScript, port 5173)** — the current
  shell at `/studio`: Story, Generate, Edit, Review, Library, Deliver, Shorts,
  Setup. It owns a studio document store, editor store, job store, and agent store,
  with TanStack Query for server reads and zod contracts for selected endpoints.
- **Legacy frontend** — the original NLE at `/`, using `timelineStore` and 7 slices.
  Frozen except for P0 fixes. `VITE_STUDIO_ENABLED=false` serves it at `/studio`.
- **Backend (Python / FastAPI, port 8000)** — inference dispatch, project/assets,
  durable GPU queue, SQLite via SQLModel, Alembic migrations, SSE. **Eleven routers
  / 113 routes** (verified 2026-11-17): projects, jobs, assets, storyboard, elements,
  shots, models, planner, review, share, depth.
- **Task engine** — `tasks/video.py`, `tasks/chained.py`, `tasks/image.py`,
  `tasks/depth.py`, `tasks/ai_parse.py`; `ModelManager` loads LTX-2; `FluxInpainter`
  wraps Flux 2; `StoryboardManager` coordinates chunks; managers handle elements,
  masks, and SAM tracking.
- **GPU scheduler** — `QueuedJob` ledger + one global semaphore + one OOM retry +
  startup recovery + pause/resume + idempotent shot jobs.
- **SAM 3 microservice (port 8001)** — health, mask prediction, detection, text
  segmentation, and video tracking sessions. Proxied through the backend.
- **MemoryManager** — mutual exclusion between heavy model slots; see
  [[../concepts/memory-slots.md]].
- **Desktop (Tauri 2)** — native Open/Save, dialog plugin, backend supervisor,
  sidecar-binary lookup. Bundled Python runtime is not shipped.

## Key Interactions

| Interaction | Protocol | Notes |
|---|---|---|
| Studio → Backend | HTTP REST | Projects/shots/elements/assets/review/share/render/model routes; one `apiUrl` resolver |
| Backend → Studio | SSE | `progress`, `complete`, `error`, `cancelled`, render and model events |
| Studio read sync | TanStack Query + local hooks | Health, models, review, renders, assets, elements, jobs |
| Studio job recovery | `GET /status/{job_id}` | Polls in-flight jobs on reconnect/mount; 404 means not running |
| Backend → SAM 3 | HTTP REST (port 8001) | Segmentation and tracking calls, off the event loop |
| Worker → LTX-2 / Flux 2 | Direct Python import | One heavy model at a time; MemoryManager gates swaps |
| Prompt enhancement | `llm.py` dispatcher | Gemma built-in or Ollama with `keep_alive: 0` |
| Render | FFmpeg subprocess | Storyboard renders from computed layout; progress parsed from FFmpeg |
| Media serving | HTTP Range requests | `206 Partial Content` for Safari-compatible mp4/mp3 |
| DB schema | Alembic | Baseline + render settings + seed lock; legacy DBs repaired/stamped |

## Current Studio vs Legacy

| Concern | Studio `/studio` | Legacy `/` |
|---|---|---|
| Store | `milimo-studio-document` + editor/job/agent stores | `milimo-timeline-storage` + 7-slice store |
| Surfaces | Story/Generate/Edit/Review/Library/Deliver/Shorts/Setup | tabbed editor, storyboard, elements, images, models |
| Board | Infinite canvas with areas/links/notes/reference/task cards | none |
| Timeline | DOM handles + isolated playback, table/keyboard path | DOM timeline |
| Render controls | Range, fit, CRF, encoder speed, audio bitrate, loudnorm, cancel | export modal |
| Review | frame comments, approvals, activity, share links | none |
| Agent | plan/approval/undo, recipes, `@Agent` | none |
| Desktop | same shell in Tauri | browser |

Both interfaces share the backend project model but have separate in-memory
documents. Switching routes does not share unsaved edits.

## Module Map (current)

**Backend** (`backend/`): `server.py` (FastAPI entry, media Range, `/events`,
`/health`), `config.py`, `database.py` (SQLModel + Alembic init), `events.py`
(SSE), `job_utils.py` (queue ledger, semaphore, retry, pause, cancellation),
`schemas.py` (Pydantic boundaries), `model_engine.py` / `memory_manager.py`,
`llm.py`, `models/flux_wrapper.py`, `managers/` (element/inpainting/tracking),
`services/` (script/ai storyboard, matcher, model registry/downloader), `tasks/`
(video, chained, image, depth, ai_parse), `routes/` (11 routers), `alembic/`
(baseline, render settings, seed lock), `utils/` (render filtergraph, captions,
paths).

**Frontend** (`web-app/src/`): `main.tsx` → `App.tsx` (Studio lazy route + legacy),
`src/studio/StudioApp.tsx`, `state/documentStore.ts` (studio document + undo),
`state/editorStore.ts`, `state/jobStore.ts`, `state/agentStore.ts`,
`data/useStudioData.ts` (hooks/contracts), `surfaces/`, `components/` (board,
timeline, inspector, command dock, lenses), `src/config.ts` (API/asset resolver),
`src/api/` (generated schema + zod contracts), `src/stores/` (legacy store/slices),
`src-tauri/` (desktop shell).

**Import edges:** `tasks/video.py` is the generation funnel; `tasks/image.py` →
`flux_wrapper.py` → `memory_manager.py`; `routes/storyboard.py` →
`ai_storyboard.py` → model engine; `routes/elements.py` → inpainting/tracking
managers → SAM 3; `storyboard/manager.py` → `element_manager.py` → Flux IP-Adapter.
Studio components read the document store and Query hooks; no component talks to
SQLite or the model layer directly.

## Deployment & Process Topology

- `./run_backend.sh` verifies environment via `backend/verify_setup.py` before
  starting Uvicorn.
- `./run_frontend.sh` starts Vite; `npm run tauri dev` hosts Studio in Tauri.
- `./run_sam.sh` starts the optional SAM 3 service.
- One GPU job at a time; queue pause stops new dispatch after the current job.
- Frontend production build is served by the backend or Tauri; API base defaults to
  the current origin outside Vite dev.

## Gotchas

- Only one LTX-2 pipeline is cached at a time; switching types unloads, clears
  caches, then reloads.
- `num_frames == 1` delegates video generation to Flux 2 image generation.
- Ollama uses `keep_alive: 0`; a 27B model otherwise sits on VRAM.
- MPS casts the VAE to float32 to prevent black output; FP8 is CUDA-only.
- Safari needs Range serving and muted-start playback.
- Do not add schema columns without an Alembic revision.
- The studio and legacy editors have separate document stores; test the route you
  changed.

## Related

- [[data-models.md]] — persistence and migrations
- [[studio-shell.md]] — implemented Studio architecture
- [[frontend-state.md]] — stores in detail
- [[nle-timeline.md]] — timeline and render truth
- [[sse-eventing.md]] — events and recovery
- [[../roadmap/implementation-plan-2026-11.md]] — status and remaining work
