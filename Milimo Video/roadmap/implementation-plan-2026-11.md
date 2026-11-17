---
type: roadmap
created: 2026-11-16
updated: 2026-11-16
tags: implementation plan correctness security nle jobs testing v2
sources:
  - wiki/systems/engineering-audit-2026-11.md
  - wiki/roadmap/v2-roadmap.md
  - wiki/research/inference-infra-2026.md
  - wiki/research/frontend-ux-workflows-2026.md
related:
  - v2-roadmap.md
  - frontend-refactor-2026-11.md
  - ../systems/engineering-audit-2026-11.md
  - ../systems/studio-shell.md
  - ../systems/nle-timeline.md
  - ../systems/data-models.md
  - ../systems/sse-eventing.md
---

# Implementation Plan — November 2026

## Purpose

Turn the [[../systems/engineering-audit-2026-11.md]] findings into a sequenced,
shippable program. This plan supersedes the **sequencing** of
[[v2-roadmap.md]], not its research or model recommendations. The v2 model swaps
remain the destination; this is the road that must exist before they can be driven.

**The rule:** no model upgrade, no agent crew, no Tauri packaging, and no 3D lane
starts until **Gate G2** (timeline truth + export fidelity) and **Gate G3** (GPU
scheduler) are green. Model quality cannot compensate for an editor that loses the
cut.

**Scope boundary:** this page owns execution order and acceptance criteria only.
Model recommendations, architecture deltas, UI/UX direction, and the new workflows
enabled remain in [[v2-roadmap.md]]; do not re-litigate them here.

## Execution Status — 2026-11-16

Working tree: `fix/implementation-plan-2026-11`. Backend
correctness gates were executed before the front-end program, per the plan's
single-engineer guidance.

| Area | Status | Evidence |
|---|---|---|
| P0 security/correctness | **Done** (P0-01–P0-11; P0-12 LICENSE/NOTICE) | `pytest`, live health, upload/media/CORS tests |
| P1 timeline truth | **Mostly done** (P1-01,03,04,05,07–10) | round-trip test, split test, revision test, tsc |
| P2 render fidelity | **Done for P2-01–P2-07; P2-12–P2-14** | unit filtergraph + real FFmpeg render test: 4.0s, video+audio |
| P2 scheduler | **Done for GPU work** (durable `QueuedJob` ledger, one retry, global semaphore, idempotency, startup resume) | `job_utils`, `QueuedJob`, `Job.shot_id`, startup recovery |
| P3 model truth | **Done** (P3-01,02,04–08) | activation routed through loader, disk paths fixed, setup check |
| P3 OOM retry | **Not started** | P3-03 deferred |
| P4 UX trust | **Partial** (P4-01,02,03,05,06) | take history/select, seed lock, save state/conflict, SSE reconnect |
| P5 tests/CI | **Partial** (pytest + vitest + CI + pre-commit + health + Playwright shell smoke) | 41 backend tests, 23 frontend tests, 1 Playwright smoke, lint 0 errors, build passes |
| Front-end F0/F1 | **Started** (shell, tokens, editor store, board canvas, list/script, timeline shell, lazy `/studio`) | `StudioApp.test` mounts the shell; tsc/lint/build pass |
| Front-end F2–F5 | **Deferred** per plan risk note | one engineer, backend gates first |

**Not adopted:** Alembic (P1-02) — a small idempotent schema repair/backfill runs
in `init_db()`; a real Alembic baseline is still required before the next schema
change. OpenAPI-generated TS types (P1-06) remain open. The durable GPU ledger
(`QueuedJob`, P2-09/G3) is implemented with startup resume and one retry.

**Verification commands**

```bash
./milimov/bin/python -m pytest        # 21 passed
./milimov/bin/python backend/verify_setup.py  # 0 failures
cd web-app && npx tsc -b && npm run lint && npm test && npm run build  # clean
curl http://127.0.0.1:8000/health     # device/queue/slots/disk
```

## Why Resequence

The v2 roadmap assumed a correct v1 foundation and planned Phase 1 model swaps
against it. The audit found the foundation does not yet hold:

- V1 ordering and conditioning do not survive save/reload.
- Export does not represent the timeline (timing, V2 compositing, audio).
- Image/inpaint jobs can run concurrently with video and OOM the machine.
- The local API is open to any web origin and has an RCE.
- Cancellation and job recovery are incomplete.

Shipping a newer LTX into that system multiplies the debugging surface. Stabilize
the contract first, then upgrade the engine.

## Guiding Principles

0. **Correctness before capability.** A smaller feature set that round-trips and
   renders truthfully beats a larger one that lies.
1. **One timeline truth.** Ordering, trims, conditioning, and track state live in
   one persisted model. UI, save, generate, and render all read the same shape.
2. **One GPU scheduler.** Every GPU task (video, image, thumbnail, inpaint,
   element visual, future lanes) enters the same queue with per-model locks.
   `MemoryManager` becomes enforcement, not advice.
3. **Contracts at the boundary.** Pydantic on write, generated OpenAPI types on
   the client, no `List[dict]` for persisted structures.
4. **Local-first includes a threat model.** Bind localhost by default, require a
   session token for non-localhost, and never execute request data.
5. **Test the fragile seams.** Save/load round-trip, render output, cancellation,
   queue restart, and security limits get automated tests before they get users.
6. **Observability is a feature.** Every job has an ID in every log line; every
   terminal state is written once and broadcast once.

## Workstreams

| WS | Name | Primary outcome | Phase |
|---|---|---|---|
| WS1 | Security hardening | No RCE, traversal, open origin, or unbounded upload | 0 |
| WS2 | Canonical timeline & persistence | Order, conditioning, trims, and track state round-trip | 0–1 |
| WS3 | Render fidelity | Export equals the cut, including V2 and audio | 1–2 |
| WS4 | Unified GPU job system | One durable queue, cancellation, restart recovery | 2 |
| WS5 | Async & service responsiveness | No event-loop blocking; SAM/inpaint async | 2–3 |
| WS6 | Model management truth | Real activate/deactivate, correct disk report, safe downloads | 3 |
| WS7 | Frontend contracts & UX | One API client, typed state, save feedback, no lost edits | 3–4 |
| WS8 | Test, CI, observability | Gates that prevent regression | 0–4, continuous |
| WS9 | v2 model & eval lane | Gated model swaps with evidence | 5+ |
| WS10 | Studio shell refactor | Board + timeline + agent over one project graph | F0–F5, parallel |

Effort scale: **S** < 1 day, **M** 2–4 days, **L** 1–2 weeks, **XL** > 2 weeks.

---

## Phase 0 — Stop the Bleeding (Days 0–3)

Goal: remove exploitable and data-destroying behavior. Ship as one small release.

| Task | Change | Files | Acceptance | Effort |
|---|---|---|---|---|
| P0-01 | Replace `eval(req.points)` with `json.loads` + coordinate validation | `routes/elements.py`, `schemas.py` | Payload `__import__('os').system(...)` returns 422; valid points still mask | S |
| P0-02 | Sanitize upload filename, enforce extension/MIME allowlist, stream with max size | `routes/assets.py` | `../../tmp/x` cannot escape; 2GB upload rejected; no `access_path` in response | S |
| P0-03 | Bind `127.0.0.1` by default; restrict CORS to configured origins; add optional session token | `server.py`, `config.py`, frontend API client | Remote bind requires `MILIMO_HOST=0.0.0.0` + token; unknown origin preflight fails | M |
| P0-04 | Safe media path (`commonpath`), streaming/`FileResponse`, Range intact | `server.py` | Traversal fuzz suite passes; 1GB file serves with flat memory | S |
| P0-05 | Fix `split_shot` datetime import, timeline serialization, new sequence index, shift following indices | `routes/projects.py`, `routes/shots.py`, `stores/slices/shotSlice.ts` | Split works with non-empty timeline; new clip appears immediately after original after reload | M |
| P0-06 | `pushStoryboardToTimeline` sets `trimOut: 0` and persists layout | `stores/slices/shotSlice.ts`, `routes/projects.py` | Pushed clips have non-zero duration and survive reload | S |
| P0-07 | Fix `prepare_shot_generation` unbound `replacement` and duplicate `. Setting:` | `storyboard/manager.py` | Empty `matched_elements` succeeds; prompt contains one setting, not N | S |
| P0-08 | Emit `cancelled` from one job runner; remove `loop` NameError; chained cancel must raise, never `break` | `tasks/video.py`, `tasks/chained.py`, `job_utils.py` | Cancel during chunk 2 → status cancelled, no stitch, exactly one SSE `cancelled` | M |
| P0-09 | Startup fails `pending` as well as `processing`; add `resume_policy` field for future | `server.py`, `database.py` | Restart leaves zero orphan `pending` (or explicit `queued` with policy) | S |
| P0-10 | Remove duplicate route handlers (`PATCH /shots/{id}`, `GET /models/disk`); add route-uniqueness test | `routes/projects.py`, `routes/shots.py`, `routes/models.py`, `tests/` | Startup test asserts no duplicate (method, path) | S |
| P0-11 | Fix chained audio trim to drop the start, not the end | `tasks/chained.py` | A/B test: audio onset stays aligned across 3 chunks | S |
| P0-12 | Add `LICENSE` + `NOTICE`; fix README claims for React version and transformer dtype | repo root, `README.md` (human-owned source — request change) | Files exist; README matches code | S |

**Gate G0:** security regression tests green; no duplicate routes; split/push/cancel
manual smoke passes. Do not tag a release without it.

---

## Phase 1 — Timeline Truth (Weeks 1–2)

Goal: one canonical timeline model that survives save, reload, split, reorder,
push, generate, and render.

### 1A. Data model

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P1-01 | Add `Shot.sequence_index` (project-scoped, unique enough for order) and a migration | Existing projects backfill by `created_at`; `GET` orders by `sequence_index` | M |
| P1-02 | Add Alembic; check in migrations; startup refuses on schema mismatch | Fresh DB and upgraded DB both boot; downgrade documented | M |
| P1-03 | Canonical `TimelineItem` (`frame_index`, `strength`, `type`, `path`) written through Pydantic | `PUT` rejects camelCase or normalizes it; round-trip keeps indices | M |
| P1-04 | Persist per-shot `height`, `fps`, `shot_type`, `index`, `status` in save payload | Full project round-trip diff is empty | S |
| P1-05 | Bump `Project.updated_at` on every mutation; add `revision` integer | Two clients cannot silently overwrite the newer revision | S |

### 1B. Frontend contracts

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P1-06 | Generate TS types from OpenAPI (or shared schema package); remove `List[dict]` from project state | `tsc` catches a renamed field | L |
| P1-07 | Central API client + asset resolver; remove 74 hardcoded localhosts | `VITE_API_URL` or `window.location` drives every call; remote browser works | M |
| P1-08 | Serialize saves (single in-flight PUT, coalesce), show Saving/Saved/Error | No out-of-order writes in a 20-edit stress test | M |
| P1-09 | Undo tracks only document edits; pause temporal during SSE/progress | One undo step per user action, not per progress event | M |
| P1-10 | Reset transient shot state on hydrate; poll 404 means not running | Reload after crash shows no permanent spinner | S |

**Gate G1:** a project with 10 ordered V1 clips, 3 V2 clips, 2 A1 clips and
non-zero conditioning indices survives: save → hard reload → identical layout,
identical payload diff, identical generation prompt inputs.

---

## Phase 2 — Render Fidelity + GPU Scheduler (Weeks 2–4)

Goal: the exported MP4 is the cut, and the machine cannot OOM from concurrent
jobs.

### 2A. Render

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P2-01 | Render from computed layout, not `start_frame` for V1 | Golden test: 3 V1 clips at 0, 4, 8s produce a 12s file with each clip visible in its window | M |
| P2-02 | V2 alpha-correct compositing; no opaque pad over V1 | A transparent PNG/alpha WebM overlays V1 without black bars | M |
| P2-03 | Include V1/V2 audio in the mix; per-clip gain/mute | Generated audio audible in preview and export | M |
| P2-04 | Send track `muted`/`hidden`/`locked` to render; apply | Muted A1 absent; hidden V1 shows black | M |
| P2-05 | Audio mix `duration=longest`; honor `trimOut`; add fades | 30s music under 10s video is not truncated; trimmed clip stops | S |
| P2-06 | Parse `ffmpeg -progress`; granular percent; cancellation | Render shows smooth percent and stops within 2s of cancel | M |
| P2-07 | Render presets (resolution, fps, codec, CRF/bitrate, audio normalize) | UI preset stored per project; output matches ffprobe | M |
| P2-08 | Golden render test fixtures (small 320×180 clips) in CI | Pixel/SSIM + duration + audio-presence assertions | L |

### 2B. Scheduler

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P2-09 | `JobQueue` table + single worker loop; all GPU tasks submit through it | Image/thumbnail/inpaint/element/video all appear in one queue | L |
| P2-10 | Per-model locks (`video`, `image`, `sam`) inside the scheduler; MemoryManager enforces | Concurrent submit of video+image never loads both; queue serializes | M |
| P2-11 | Restart recovery: `queued` resumes, `running` becomes `interrupted` with retry policy | Kill -9 mid-job; restart resumes or marks retryable once | M |
| P2-12 | Cancellation state machine (`queued→cancelled`, `running→cancelling→cancelled`) with one broadcast | Cancel in queue, in denoise, and during render all converge | M |
| P2-13 | Idempotency: reject duplicate active job per shot; client sends request ID | Double-click yields one job | S |
| P2-14 | Move blocking inference to threads/processes; SAM via `httpx.AsyncClient` | `/status` responds <100ms during inpaint/tracking | M |

**Gate G2:** golden export matches the timeline; no audio loss; V2 compositing
works; 10 mixed jobs submitted at once never exceed one heavy model and all reach
a terminal state.

**Gate G3:** kill/restart during a render and during a video job leaves a
recoverable queue with no zombie `processing`/`pending`.

---

## Phase 3 — Model Management & Service Truth (Weeks 4–5)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P3-01 | Route activation through `model_loader.activate_base_model` / `deactivate_base_model`; hold the manager lock | Activate loads; deactivate frees; memory endpoint reflects reality | M |
| P3-02 | VRAM/RAM preflight before activation and before generation; block with actionable error | 32GB Mac refuses a model that cannot fit | S |
| P3-03 | OOM retry ladder: lower resolution / fewer frames / sequential offload, once | A forced OOM retries at the next safe preset and reports it | M |
| P3-04 | One `/models/disk` schema; correct LTX/SAM/Flux paths | Dashboard matches `du` within tolerance | S |
| P3-05 | Downloads always through the priority queue; resume via `.incomplete`; remove deprecated arg; cancel queued | Two clicks → one download; kill/resume works; cancel queued returns 200 | M |
| P3-06 | Pin `requirements` (torch/transformers/diffusers/hf_hub) and add `requests`, `huggingface_hub`; add dev extras | Fresh venv installs and boots | M |
| P3-07 | `verify_setup` command: imports, ffmpeg/ffprobe, model paths, writable dirs, device | `./run_backend.sh` fails fast with the exact missing item | S |
| P3-08 | Remove debug artifacts from hot path; add `MILIMO_DEBUG` gate; request/job IDs in logs | No prompts written to CWD by default | S |

---

## Phase 4 — Frontend Finish & UX Trust (Weeks 5–6)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P4-01 | Take history per shot (job lineage, compare, restore) | Regenerate keeps both takes; restore updates shot atomically | L |
| P4-02 | Seed lock/randomize toggle; deterministic re-run | Same seed + params reproduces | S |
| P4-03 | Save state, conflict banner, retry; flush edits on unmount | No lost storyboard edits in navigation test | M |
| P4-04 | Storyboard/timeline virtualization or canvas layer if clip count >200 | 500 clips at 60fps scrub | L |
| P4-05 | SSE reconnect re-polls active jobs + heartbeat timeout | Kill network 10s during completion; UI recovers | S |
| P4-06 | Empty/error/loading states for every panel; actionable OOM/queue messages | No silent failures | M |
| P4-07 | Accessibility pass on keyboard, focus, labels, contrast | Keyboard-only edit of a 3-shot sequence | M |

---

## Phase 5 — Testing, CI & Observability (Starts Day 0, Gate at Week 6)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| P5-01 | `pytest` suite: schemas, save/load round-trip, render graph, queue, cancellation, security limits | >80% on changed modules; no network/GPU required for unit tests | L |
| P5-02 | `vitest` store tests: ordering, conditioning, split, push, undo | Contract tests catch snake/camel regressions | M |
| P5-03 | Playwright smoke: create project → add clips → save/reload → export → verify duration | Shell/board/review/setup smoke runs headless with a stubbed API; full create→export flow still pending | M |
| P5-04 | CI: ruff + pytest + tsc + eslint (fix the 117 problems or baseline them explicitly) + vitest | PR cannot merge red | M |
| P5-05 | Pre-commit hooks: ruff format/check, tsc, no large files | Local parity with CI | S |
| P5-06 | Structured logs with job/request IDs; `/health` includes device, model slots, queue depth, disk | One curl explains a stuck studio | S |
| P5-07 | Error budget dashboard (failed jobs by type, OOM count, render failures) | Weekly review possible | M |

---

## Front-End Refactor Program (Parallel, Contract-Gated)

The current UI cannot be fixed by incremental polish: its store, contracts, and
layout encode the same defects the audit found. The full design is in
[[frontend-refactor-2026-11.md]] and [[../systems/studio-shell.md]]; the external
evidence is in [[../research/frontend-video-platforms-2026.md]].

The program is a **strangler refactor** behind `/studio` and per-surface flags:

| Stage | Scope | Depends on | Effort |
|---|---|---|---|
| **F0** | Design tokens, shell, API client, state skeleton, tests, a11y | Can start now | 2–3 weeks |
| **F1** | Story: script, board, shot list, areas/links, LOD/minimap/list | G1 | 3–4 weeks |
| **F2** | Edit: stage, timeline, tools, audio, captions, inspector, render | G2 | 3–4 weeks |
| **F3** | Generate: shot workspace, ingredients, takes, compare, queue | G3 | 2–3 weeks |
| **F4** | Agent command dock, review/version stacks, library, deliver | G3 + review APIs | 3–4 weeks |
| **F5** | Shorts workspace, pipeline/blocking/3D lenses, Tauri, collaboration | F2/F4 | 4–6 weeks |

**Key design decisions**

- **Board + timeline are peers**, not modes that replace each other. Both read
  the same `Shot`/`Take` graph; `sequence_index` defines order, never canvas x/y.
- **Takes are immutable.** Every generation creates a take; the shot selects one;
  review compares them.
- **The agent plans before spending.** Model, objects, time/VRAM, diff preview,
  approval; applied changes are undoable patches.
- **Canvas borrows Kubona's interaction vocabulary** (Apache 2.0): camera, LOD,
  culling, areas, links, pins, minimap, list view, command palette, inspector,
  copilot drawer, split layout/view saves. Borrow patterns, not the quantum domain.
- **No God store.** Document / editor / job / agent stores; TanStack Query for
  server state; Immer patches for undo; playback isolated.
- **No hardcoded localhost.** One API/SSE client, generated types, Zod validation.

**Front-end gates:** F0 shell; F1 story round-trip; F2 export match; F3 take/queue
truth; F4 review/agent truth; F5 shorts/lenses.

**Effort:** 12–14 weeks with two front-end engineers and part-time design; 20–24
weeks with one. Do not run a single engineer on both backend correctness and this
refactor.

## Phase 6 — Resume v2 (After G2/G3)

Only after the studio is truthful:

1. **Eval harness first** — VBench-2.0 subset + FFmpeg sanity + blind A/B UI, as
   the v2 roadmap requires. No swap without scores.
2. **Video slot** — LTX-2.5 22B behind a feature flag; keep LTX-2 19B pinned.
3. **Image slot** — Z-Image Turbo + Qwen-Image-Edit-2511 split; keep Flux fallback.
4. **LLM slot** — Qwen3.x for rewrite and Qwen3-VL for critique; keep Ollama
   `keep_alive` discipline and move AI storyboard parsing off the 19B pipeline.
5. **Architecture** — FP8 budgets on NVIDIA, MLX spike on Mac, sequential offload.
6. **UX/agentic** — React Compiler, canvas timeline, copilot with diff/approval,
   MCP server, Tauri 2 sidecar.

The v2 roadmap's model recommendations stand. Its Phase 0 is now **complete** for
the eight documented contradictions (see the audit's verification table); its
Phase 1 start moves behind G2/G3.

---

## Quality Gates

| Gate | Criteria | Blocks |
|---|---|---|
| **G0** | Security tests green; no duplicate routes; split/push/cancel smoke | Any release |
| **G1** | Save → reload → identical project, including order/conditioning/trims | Generation work that depends on layout |
| **G2** | Golden export equals the cut: timing, V2 alpha, audio, track state | Calling the product an NLE |
| **G3** | One GPU queue; restart recovery; cancellation converges | Multi-job use, batch generation |
| **G4** | Model activation real; fresh install boots; downloads resume | Model upgrades |
| **G5** | CI green; Playwright smoke; lint/type gates | v2 work |

## Test Matrix (Minimum)

| Layer | Test | Tool |
|---|---|---|
| Schema | `TimelineItem` accepts only canonical keys; rejects traversal filenames | pytest + Hypothesis |
| Persistence | 10-shot project round-trip with order, trims, conditioning, shot type | pytest + temporary SQLite |
| Render | 3 V1 + 1 V2 alpha + 2 A1: duration, visible windows, audio presence | pytest + ffprobe/SSIM |
| Queue | Submit 10 mixed jobs; assert one heavy model, all terminal | pytest + fake executor |
| Cancellation | Cancel queued/running/render; assert one event and terminal DB state | pytest + TestClient |
| Security | `eval` payload, `../` upload, foreign Origin, 2GB limit | pytest |
| Frontend | `computeTimelineLayout`, split, push, undo, save serialization | vitest |
| E2E | Create → edit → save/reload → export | Playwright |
| Perf | 500 clips scrub; 1GB range serve memory | Playwright + psutil |

## Sequencing & Effort

Assumes one senior full-stack engineer plus part-time review:

- **Week 1:** Phase 0 + P1-01–P1-05. Security and data model.
- **Week 2:** P1-06–P1-10 + P2-01–P2-05. Contracts and render basics.
- **Week 3:** P2-06–P2-14. Golden render, scheduler, async services.
- **Week 4:** Phase 3 + P5-01–P5-04. Model truth, CI, tests.
- **Week 5:** Phase 4. Takes, seed lock, save UX, SSE recovery.
- **Week 6:** Hardening, Playwright, performance, release candidate.
- **Week 7+:** Eval harness, then v2 model swaps.

Total to a trustworthy backend: **~6 weeks** for one engineer; **~3–4 weeks** with
two working in parallel (backend/queue + frontend/contracts). The full studio
refactor adds **12–14 weeks** with two front-end engineers (F0–F4), starting F0
immediately and F1 after G1. The v2 model lane adds 4–6 weeks after G2/G3.

**Recommended staffing:** one backend/correctness engineer + two front-end
engineers + part-time design. One engineer cannot carry both tracks; if only one
engineer is available, ship the backend gates first and keep the legacy UI until
F1 can start.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Migration breaks existing projects | Back up `milimovideo.db` automatically before migration; dry-run + rollback; test on a copy of the live DB |
| Scheduler rewrite stalls generation | Land the queue behind a feature flag; keep the current semaphore path until parity tests pass |
| Render golden tests are flaky across FFmpeg versions | Pin FFmpeg in CI; assert duration/stream/SSIM tolerances, not exact bytes |
| Scope creep back into v2 | Gates are contractual; no Phase 6 task starts before G2/G3 |
| Security token breaks EventSource | Token via query parameter + `SameSite=Strict` cookie fallback; test Safari |
| MPS bf16 instability resurfaces | Keep CPU VAE decode; add a dtype smoke render on MPS to CI-on-hardware |
| Single-maintainer bandwidth | Phase 0 and 1 are the minimum viable trust release; defer the studio refactor rather than half-build it |
| Front-end big-bang rewrite stalls | Strangler `/studio` route + per-surface flags; legacy fallback; cut over only after F2/F4 |
| Dual UI maintenance | Freeze legacy except P0; one release maximum; no feature work in both shells |
| Canvas/timeline perf regressions | 500-card/200-clip budgets in CI; culling/LOD from day one; no per-frame React |

## Decisions Needed

1. **API exposure:** localhost-only by default, or LAN with a token? (Recommend:
   localhost default, token for LAN.)
2. **Database:** stay SQLite + Alembic, or move to Postgres now? (Recommend:
   SQLite for v1; the queue is single-writer and local-first.)
3. **Render engine:** FFmpeg filtergraph for v1, or a compositor (e.g., MLT/
   GStreamer) later? (Recommend: fix FFmpeg now; revisit at G2.)
4. **Takes:** store every generation as an immutable Asset + a `ShotTake` join, or
   keep one URL per shot plus gallery? (Recommend: `ShotTake`.)
5. **Model upgrades:** keep LTX-2 19B pinned until the eval harness exists?
   (Recommend: yes.)
6. **Packaging:** Tauri after G5, or browser-only for v1? (Recommend: after G5.)

## Definition of Done (Trustworthy v1)

- A 10-shot project can be built, reordered, trimmed, conditioned, saved, reloaded,
  and exported with frame-accurate timing, correct V2 compositing, and audible
  generated audio.
- No GPU task can run outside the scheduler; killing the app mid-job is recoverable.
- No request can execute code, escape a directory, or be driven by a foreign origin.
- A fresh clone can install, verify, and boot with one documented command.
- CI blocks regressions in schemas, persistence, render, queue, cancellation, and
  security.
- The studio shell makes the board, shot, and cut one project: a 10-shot film can
  be written, boarded, generated, edited, reviewed, and exported without leaving
  the app; takes are immutable; the agent plans before spending.
- 500 board cards and 200 timeline clips hold 60fps; every canvas has a list view;
  no hardcoded localhost; generated contracts.
- The wiki's audit page is updated to mark each P0/P1 as fixed, with the commit.

## Related

- [[../systems/engineering-audit-2026-11.md]] — evidence and severity
- [[v2-roadmap.md]] — model and architecture destination
- [[../systems/nle-timeline.md]] — timeline semantics
- [[../systems/data-models.md]] — persistence contracts
- [[../systems/sse-eventing.md]] — job events
- [[../research/inference-infra-2026.md]] — slot and quantization context
- [[../research/frontend-ux-workflows-2026.md]] — React/canvas/copilot context
