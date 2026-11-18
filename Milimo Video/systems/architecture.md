---
type: systems
created: 2026-10-09
updated: 2026-11-18
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
microservice on port 8002, an optional ComfyUI server (port 8001 by default), and a Tauri desktop wrapper that hosts the same shell.

## Components

- **Studio frontend (React 19.2 / Vite / TypeScript, port 5173)** — the current
  shell at `/studio`: Story, Generate, Edit, Review, Library, Deliver, Shorts,
  Setup. It owns a studio document store, editor store, job store, and agent store,
  with TanStack Query for server reads and zod contracts for selected endpoints.
- **Legacy frontend** — the original NLE at `/`, using `timelineStore` and 7 slices.
  Frozen except for P0 fixes. `VITE_STUDIO_ENABLED=false` serves it at `/studio`.
- **Backend (Python / FastAPI, port 8000)** — inference dispatch, project/assets,
  durable GPU queue, SQLite via SQLModel, Alembic migrations, SSE. **Thirteen routers
  / 145 operations** (verified 2026-11-18): projects, jobs, assets, storyboard,
  elements, shots, models, planner, review, share, depth, settings, comfy.
- **Task engine** — `tasks/video.py`, `tasks/chained.py`, `tasks/wan.py`,
  `tasks/image.py`, `tasks/comfy.py`, `tasks/depth.py`, `tasks/ai_parse.py`;
  `ModelManager` loads LTX-2; `FluxInpainter` wraps Flux 2; `WanEngine` wraps the
  low-VRAM Wan 2.2 lane; `LTX25Engine` runs gated diffusers LTX-2.5; `ComfyClient`
  runs external ComfyUI workflows;
  `StoryboardManager` coordinates chunks; managers handle elements, masks, and
  SAM tracking.
- **LLM plane** — `services/llm/`: persisted settings with provider **instances**
  (`store.py`, `instances.py`), wire adapters with capability parsing, images, and
  streaming (`providers.py`), role execution with preflight/fallback/cancel
  (`runtime.py`), loopback discovery (`discovery.py`), role eval (`eval.py`,
  `fixtures.py`), cloud budget ledger (`budget.py`), profiles (`profiles.py`), and
  AES-GCM credential sealing (`secretbox.py`). `backend/llm.py` is a compatibility
  facade. See [[llm-local-first-next-2026-11]].
- **GPU scheduler** — `QueuedJob` ledger + one global semaphore + one OOM retry +
  startup recovery + pause/resume + idempotent shot jobs. Slots are being upgraded
  to a measured memory budget with external-engine accounting, leases, TTL eviction,
  and pressure response; see [[memory-management-audit-2026-11]].
- **SAM 3 microservice (port 8002)** — health, mask prediction, detection, text
  segmentation, and video tracking sessions. Proxied through the backend.
- **ComfyUI bridge** — `services/comfy.py`, `tasks/comfy.py`, `routes/comfy.py`:
  submit API-format workflows, relay websocket/polling progress, stream outputs
  into project assets and shots. ComfyUI is an external GPU engine with a
  declared `comfy` memory slot. See [[comfyui-bridge]].
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
| Backend → SAM 3 | HTTP REST (port 8002) | Segmentation and tracking calls, off the event loop |
| Backend → ComfyUI | HTTP REST + WebSocket | `/prompt`, `/history`, `/view`, `/queue`, `/free`, `/interrupt`; progress on `/ws` |
| Worker → LTX-2 / Flux 2 | Direct Python import | One heavy model at a time; MemoryManager gates swaps |
| Prompt enhancement | `llm.py` → role runtime | `prompt_video`/`prompt_image` resolve to persisted provider/model hops with ordered fallback |
| LLM settings | `GET/PUT/PATCH /settings/llm` | Redacted snapshot, revision CAS, air-gap patch; writes persist in `app_setting` |
| Provider health | `/settings/llm/{models,discover,verify,warm,health}` | Model catalog with context/vision/chat flags, loopback scan, live verify, warm-up, saved-config test |
| LLM eval / budget | `/settings/llm/{eval,eval/apply,profiles,spend}` | Role fixtures with pass rate/p50, recommended plan, staged profiles, cloud spend ledger |
| Prompt provenance | `Job.params_json.llm` | role, provider, model, latency, token usage |
| Render | FFmpeg subprocess | Storyboard renders from computed layout; progress parsed from FFmpeg |
| Media serving | HTTP Range requests | `206 Partial Content` for Safari-compatible mp4/mp3 |
| DB schema | Alembic | Baseline + render settings + seed lock + asset thumbnail + app settings; legacy DBs repaired/stamped |

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
`/health`, opt-in reload), `config.py`, `database.py` (SQLModel + Alembic init),
`events.py` (SSE with `seq`/replay), `job_events.py` (typed `job.*` envelope and
the single terminal emitter), `job_utils.py` (durable ledger, GPU semaphore,
priority + same-slot ordering, bounded memory waits, retry, pause, cancellation),
`schemas.py` (Pydantic boundaries), `model_engine.py` / `memory_manager.py`
(residency, leases, admission), `services/memory/` (`models`, `costs`, `engines`,
`budget`, `sampler`), `llm.py` (facade), `models/flux_wrapper.py`, `managers/`
(element/inpainting/tracking), `services/` (script/ai storyboard, matcher, model
registry/downloader, `llm/` settings+providers+runtime+discovery), `tasks/`
(video, chained, image, depth, ai_parse), `routes/` (13 routers incl. `memory.py`),
`alembic/` (baseline, render settings, seed lock, asset thumbnail, app settings,
job progress), `benchmark_models.py` (M5 lane baselines), `utils/` (render
filtergraph, captions, paths).

**Frontend** (`web-app/src/`): `main.tsx` → `App.tsx` (Studio lazy route + legacy),
`src/studio/StudioApp.tsx`, `state/documentStore.ts` (studio document + undo),
`state/editorStore.ts`, `state/jobStore.ts` (active/history/unread),
`state/agentStore.ts`, `data/useStudioData.ts` (hooks/contracts, bootstrap,
preflight), `events/jobBridge.ts` (job→shot/toast reconciler),
`notifications/preferences.ts` (toasts/OS/sound/project mute), `surfaces/`,
`components/` (board, timeline, inspector, command dock, lenses, toasts, activity,
memory), `src/config.ts` (API/asset resolver), `src/api/` (generated schema + zod
contracts), `src/stores/` (legacy store/slices), `src-tauri/` (desktop shell with
title/badge/notify commands).

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
- One GPU job at a time; dispatch is priority-aware (interactive > bulk > parse)
  with same-slot grouping; queue pause stops new dispatch after the current job.
- Memory admission runs before every heavy load. A transient refusal becomes a
  bounded “waiting for memory” retry; a permanent one offers/applies a smaller
  preset before any allocation.
- Uvicorn reload is opt-in (`MILIMO_RELOAD=1`); normal runs do not watch the repo.
- The SAM 3 image model is lazy (loads on first segmentation request).
- Frontend production build is served by the backend or Tauri; API base defaults to
  the current origin outside Vite dev.

## Gotchas

- Only one LTX-2 pipeline is cached at a time; switching types unloads, clears
  caches, then reloads.
- `num_frames == 1` delegates video generation to Flux 2 image generation.
- Ollama defaults to `keep_alive: 0`; a 27B model otherwise sits on VRAM. The
  active role plan, not `config.LLM_PROVIDER`, chooses the provider now. HTTP
  enhancement runs before the video pipeline loads and releases every local role
  model afterwards; chunk-0 HTTP enhancement is now marked pre-enhanced so the
  chained task does not overlap the LTX load.
- Gemma is in-process only: a role that resolves to Gemma needs the loaded LTX
  text encoder, so `prompt_image` must use an HTTP provider.
- Cloud providers are unreachable unless `MILIMO_LLM_CLOUD=1` **and** the air-gap
  switch is off; both checks live in `store.plans()`.
- MPS casts the VAE to float32 to prevent black output; FP8 is CUDA-only.
- Safari needs Range serving and muted-start playback.
- Do not add schema columns without an Alembic revision.
- The studio and legacy editors have separate document stores; test the route you
  changed.
- A job must emit stages and exactly one terminal event through `job_events`; do
  not add ad-hoc `event_manager.broadcast` calls for job state.
- Any new heavy model must call `memory_manager.admit_job` / `prepare_for` and be
  included in the cost registry, or it bypasses the budget.
- The physical-free and free-swap checks are separate from the MPS ceiling; swap
  occupancy alone is historical, not live pressure.
- `useStudioProjectBootstrap` treats id `default` as a local sentinel and opens the
  newest real server project; do not persist `default` as a real choice.

## Related

- [[data-models.md]] — persistence and migrations
- [[studio-shell.md]] — implemented Studio architecture
- [[frontend-state.md]] — stores in detail
- [[nle-timeline.md]] — timeline and render truth
- [[sse-eventing.md]] — typed events, replay, and recovery
- [[../concepts/memory-slots.md]] — slots and the measured budget
- [[../roadmap/implementation-plan-2026-11.md]] — status and remaining work
