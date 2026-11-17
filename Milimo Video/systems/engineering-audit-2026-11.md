---
type: systems
created: 2026-11-16
updated: 2026-11-16
tags: audit security correctness persistence jobs nle render quality
sources:
  - backend/
  - web-app/src/
  - docs/CODEBASE_ANALYSIS.md
  - docs/user_guide.md
related:
  - architecture.md
  - data-models.md
  - nle-timeline.md
  - sse-eventing.md
  - ../pipelines/chained-generation.md
  - ../pipelines/inpainting.md
  - ../roadmap/implementation-plan-2026-11.md
  - ../roadmap/v2-roadmap.md
---

# Engineering Audit — November 2026

## Purpose

A code-level audit of Milimo Video's current `main` (commit `807985e`, plus the
uncommitted wiki/docs) to answer one question: **can a filmmaker trust this studio
with a real project today?** The answer is no, not yet — not because the models are
weak, but because edits do not reliably round-trip, exports do not faithfully
represent the timeline, and the local API is exposed without a threat model.

This page records the evidence, the severity-ranked gaps, and what is genuinely
solid. The executable response lives in
[[../roadmap/implementation-plan-2026-11.md]]. No code was changed by this audit.

## Method & Scope

- Static read of every first-party backend module (`backend/*.py`, `routes/`,
  `tasks/`, `managers/`, `services/`, `storyboard/`, `utils/`), the frontend store
  and the components that drive generation, playback, and export.
- `python3 -m compileall` — passes (runtime `NameError`s are invisible to it).
- `npx tsc -b` — passes.
- `npm run lint` — **fails**: 117 problems (101 errors, 16 warnings), mostly
  `no-explicit-any`, so the lint gate is currently non-existent.
- FastAPI route table imported and enumerated: **93 routes**, two duplicate paths.
- Live SQLite inspection (`backend/milimovideo.db`): 3 projects, 14 shots,
  42 completed jobs, 2 jobs permanently `pending`; every V1 shot has
  `start_frame = 0`; every `shot.timeline` is empty.
- Not run: GPU inference, a real render, SAM tracking. Findings that depend on
  those paths are marked as code-path evidence, not observed runtime failures.

## Executive Verdict

The engine room is real: the LTX-2/Flux 2 wrappers, the MemoryManager intent, the
SSE transport, the storyboard engine, and the SAM 3 isolation are all substantial.
But the **studio layer around the engines is not yet trustworthy**:

1. **The timeline is not the source of truth.** V1 order is never persisted; the
   database returns shots by `created_at`, and export positions every V1 clip at
   `start_frame = 0`. Reordering clips is lost on reload, and exports overlay all
   V1 clips on top of each other.
2. **Conditioning does not survive a save.** The frontend writes `frameIndex`, the
   backend stores it verbatim, and the loader reads `frame_index`. Every image/video
   conditioning index becomes `0` after reload.
3. **Exports are not the cut.** V1/V2 embedded audio is discarded, V2 overlays are
   padded with opaque black over V1, and hidden/muted tracks are ignored.
4. **Security is a drive-by risk.** `eval()` on request data, unsanitized upload
   filenames, and `CORS: *` on a `0.0.0.0` API with no auth mean any web page the
   user visits can drive their local studio.
5. **Cancellation and job recovery are partly fictional.** Video cancellation never
   emits the `cancelled` event; chained cancellation can stitch a partial film and
   report success; pending jobs are never recovered on restart.
6. **The model manager UI lies.** "Activate" marks a model active without loading
   it; the real loader methods are never called.

**Recommended posture:** freeze model-upgrade work (v2 Phase 1) until the P0 fixes
and the timeline/render truth work land. Model quality is not the current moat
problem; correctness is.

## Remediation Status — 2026-11-16

| Finding | Status | Change |
|---|---|---|
| SEC-01 `eval` RCE | **Fixed** | JSON point parser, validation, 422 tests |
| SEC-02 upload traversal/size | **Fixed** | basename sanitization, extension allowlist, streaming 1MB chunks, 2048MB cap |
| SEC-03 open CORS/no auth | **Fixed** | loopback default, origin allowlist, optional bearer token, preflight 403 test |
| SEC-04 media traversal/memory | **Fixed** | `commonpath`, streaming Range, `FileResponse` |
| DATA-01 V1 order | **Fixed** | `Shot.sequence_index`, save order, GET order, backfill, render uses computed V1 starts |
| DATA-02 conditioning key drift | **Fixed** | canonical `TimelineItem`, camelCase normalization, list on read |
| DATA-03 split shot | **Fixed** | datetime import, timeline serialization, sequence/scene index shift |
| DATA-04 push trimOut | **Fixed** | `trimOut: 0`, persisted patch + save |
| DATA-05 no migrations | **Fixed** | Alembic baseline + startup upgrade; legacy DBs repaired/stamped; migration tests |
| DATA-06 save omissions | **Fixed** | height/fps/shotType/index/sequence_index/revision persisted |
| DATA-07 destructive storyboard merge | **Fixed (v1)** | pre-commit archive persisted/exposed; Script UI requires confirmation and can download backup |
| DATA-09 pending job recovery | **Fixed** | durable `QueuedJob` ledger; queued resumes, running retried once, undispatched pending fails |
| GEN-01 prompt `replacement` bug | **Fixed** | scoped replacement, no duplicate `. Setting:` |
| GEN-02 export audio | **Fixed** | V1/V2 embedded audio mixed; preview still muted by default |
| GEN-03 GPU queue bypass | **Fixed** | image/thumbnail/inpaint/element all use durable `QueuedJob` + `gpu_job_wrapper` |
| GEN-04 cancellation | **Fixed** | direct `cancelled` broadcast; chained cancel raises, partial stitch blocked |
| GEN-05 chained audio trim | **Fixed** | trims start, matching video |
| GEN-06 seed randomization | **Fixed** | seed locked; `randomize_seed=true` on Regen |
| GEN-07 AI parse blocks loop | **Fixed** | AI parse is a durable queued job; model load/parse run off-request; result JSON is polled and committed |
| JOB-01 no durable queue | **Fixed** | `QueuedJob` ledger + startup resume + one retry |
| JOB-06 duplicate routes | **Fixed** | removed duplicate handlers; route uniqueness test |
| MOD-01 activation no-op | **Fixed** | routed through `activate_base_model`/`deactivate_base_model` |
| MOD-02 disk paths | **Fixed** | LTX-2/, backend/models/flux2, backend/models/sam3 |
| MOD-03 duplicate `/models/disk` | **Fixed** | one rich schema, registered before `{model_id}` |
| MOD-04 downloader bypass | **Fixed** | always queued, deprecated arg removed, `.incomplete` detection, cancel queued |
| OPS-03 dependencies | **Fixed** | requests/hf_hub/pytest/alembic/httpx/ruff added |
| OPS-04 broken env | **Fixed** | `config.setup_paths()` at import + `verify_setup.py` + fail-fast run script |
| OPS-05 no LICENSE | **Fixed** | Apache-2.0 `LICENSE`, `NOTICE` |
| OPS-06 debug artifacts | **Fixed** | gated behind `MILIMO_DEBUG` |
| Take/version history | **Fixed (v1)** | immutable takes from completed jobs; `GET /shots/{id}/takes` + select-take repoints the shot |
| Review notes | **Fixed (v1)** | frame comments + per-shot approvals persisted in `review_item`; latest approval gates the shot |
| Captions | **Fixed (v1)** | cues persist, SRT/VTT import, optional burn; render writes ASS |
| API contracts | **Fixed (v1)** | typed ProjectState/ShotState; generated OpenAPI TS + Zod checks; CI drift check |
| Observability | **Fixed (v1)** | X-Request-ID + timing logs; metrics endpoint + Setup error-budget panel |
| Performance | **Fixed (v1)** | 500-card board culling/pan ~58fps; 200-clip timeline culling/scroll ~275fps; headless specs committed |
| Accessibility | **Fixed (v1)** | axe WCAG A/AA zero serious/critical; skip link; keyboard timeline editing |
| NLE-01 timing | **Fixed** | render computes V1 starts from sequence order |
| NLE-02 V2 black | **Fixed** | transparent pad + RGBA + alpha overlay |
| NLE-03 track state | **Fixed** | `track_states` sent from ExportModal and applied |
| NLE-04 amix truncation | **Fixed** | `duration=longest`, output `-t` |
| NLE-05 audio trimOut | **Fixed** | GlobalAudioManager duration honors trims |
| NLE-07 render cancel | **Fixed** | active_jobs registration + terminate/kill |
| NLE-08 progress | **Fixed** | `-progress pipe:1` parsing |

Open/partial items are tracked in
[[../roadmap/implementation-plan-2026-11.md]].

## Scorecard

| Area | Grade | One-line verdict |
|---|---|---|
| Security & isolation | **D** | RCE, path traversal, open CORS, no auth, no upload limits |
| Data integrity | **D** | V1 order and conditioning indices do not round-trip; no migrations |
| Generation orchestration | **C−** | Strong pipelines, but image/inpaint jobs bypass the GPU queue |
| NLE playback & export | **F** | Export timing, V2 compositing, and audio are all wrong |
| Job lifecycle & SSE | **C** | Good transport; cancellation/replay/recovery are incomplete |
| Frontend state & contracts | **C** | Typed façade over loosely coupled snake/camel contracts |
| Model management | **D** | Registry is useful; activation is a no-op; disk paths are wrong |
| Tests, CI, packaging | **F** | No tests, no CI, no lockfile, no LICENSE file, lint fails |

## P0 Blockers

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| SEC-01 | `eval()` on `InpaintRequest.points` | `backend/routes/elements.py:119` | Remote code execution from any HTTP client that can reach the API |
| SEC-02 | Upload filename is joined without sanitization or size limit | `backend/routes/assets.py:44-48` | Arbitrary file write outside the project via `../`; disk exhaustion |
| SEC-03 | `CORS allow_origins=["*"]`, no auth, binds `0.0.0.0` | `backend/server.py:106-137`, `server.py:248` | Any website can read/write the local studio (CSRF / DNS rebinding) |
| SEC-04 | Media traversal uses `startswith`, full file read | `backend/server.py:171-234` | Sibling-directory reads; multi-GB memory spike per request |
| DATA-01 | V1 order is not persisted; export uses `start_frame` | `routes/projects.py:94, 465-470`; `render_utils.py:139`; live DB all V1 `start_frame=0` | Reorder lost on reload; every V1 clip overlays at `t=0` in export |
| DATA-02 | Conditioning index serialized as `frameIndex`, read as `frame_index` | `projectSlice.ts:97, 208`; `projects.py:267-283`; `manager.py:372` | All conditioning collapses to frame 0 after save/reload |
| DATA-03 | Split shot is broken end-to-end | `projects.py:587` (`datetime` not imported); `shotSlice.ts:135` (`.map` on a JSON string) | Split returns 500 or crashes client; new shot also gets the old `index` |
| DATA-04 | "Push to timeline" sets `trimOut = numFrames` | `shotSlice.ts:574` | Zero-length clips on V1; export renders nothing |
| GEN-01 | `prepare_shot_generation` can raise `UnboundLocalError`, then duplicates `. Setting:` | `storyboard/manager.py:345-346` | Shot generation fails or prompt is polluted with wrong setting text |
| GEN-02 | Export discards generated audio; preview is muted by default | `render_utils.py:75, 116`; `CinematicPlayer.tsx:142` | LTX audio is never in the export; preview only unmutes after a global user gesture and has no clip-level audio control |
| GEN-03 | Image, thumbnail, inpaint and element jobs bypass the GPU semaphore | `storyboard.py:426`; `jobs.py:268`; `elements.py:92,161`; `job_utils.py:41` | Concurrent Flux + LTX loads; OOM and corrupted generations |
| GEN-04 | Video cancellation never broadcasts `cancelled`; chained cancel can `break` then stitch | `tasks/video.py:785-790`; `tasks/chained.py:252` | UI stuck on "Cancelling"; partial film can be reported complete |
| GEN-05 | Chained audio trims the end while video trims the start | `tasks/chained.py:343-350` | ~1s audio desync per chunk |

## Detailed Findings

### Security & Isolation

- **SEC-01 — RCE through `eval`.** `POST /edit/inpaint` passes
  `eval(req.points)` to mask generation. Any expression executes with the backend's
  privileges. Replace with `json.loads`, validate coordinate shape, and reject
  non-numeric values.
- **SEC-02 — Upload path traversal and unbounded writes.** `file.filename` is
  concatenated after an 8-char UUID; `../` segments escape the target directory.
  There is also no size cap, no MIME/extension allowlist, and
  `file.content_type.startswith(...)` raises if the header is absent
  (`assets.py:66`). Use `Path(name).name`, an allowlist, `Content-Length`/streaming
  limits, and magic-byte checks.
- **SEC-03 — Open local API.** `allow_origins=["*"]` with
  `allow_credentials=False` still lets any origin read responses. A malicious page
  can enumerate projects, read generated prompts, start renders/downloads, delete
  assets, or set `HF_TOKEN`. Bind `127.0.0.1` by default, restrict origins to the
  Vite/Tauri origins, and add a per-session bearer token (EventSource can carry it
  as a query token).
- **SEC-04 — Media serving.** `full_path.startswith(normpath(base_dir))` accepts
  sibling paths such as `.../projects_backup`. Use
  `os.path.commonpath([base, full]) == base`. The no-Range branch reads the entire
  file into RAM and copies it into a `Response`; use `FileResponse`/streaming with
  `Range`, and set `Content-Disposition` only on download routes.
- **SEC-05 — Tracking persistence path.** `save_tracking_results` hashes
  `video_path` (safe), but `load_tracking_results` joins raw `session_id` into a
  path (`elements.py:453-455`). Constrain to `[a-zA-Z0-9_-]`.
- **SEC-06 — Information disclosure.** Upload returns `access_path`, an absolute
  server path (`assets.py:75`). Return only the web URL; keep filesystem paths
  server-side.

### Persistence & Data Integrity

- **DATA-01 — V1 order is not a persisted property.** The frontend's
  `computeTimelineLayout()` derives V1 order from array order
  (`timelineUtils.ts:26-42`), and `reorderShots()` only reorders that array
  (`shotSlice.ts:171-181`). `PUT /projects/{id}` writes fields but no order
  (`projects.py:254-295`), and `GET` returns `ORDER BY created_at`
  (`projects.py:94`). The live DB confirms every V1 shot has `start_frame=0`.
  Export then treats `start_frame` as absolute
  (`projects.py:465-470`), so every V1 clip is overlaid at `t=0`. **Fix:** add a
  `sequence_index` (project-level) or a `TimelineItem` table, persist it, order all
  reads by it, and have render compute V1 starts magnetically, not from
  `start_frame`.
- **DATA-02 — Two naming systems for conditioning.** Frontend type uses
  `frameIndex`; `ShotConfig.TimelineItem` uses `frame_index`; `saveProject()`
  sends `frameIndex` (`projectSlice.ts:94-99`), stores the JSON verbatim
  (`projects.py:267-283`), and `loadProject()` reads `frame_index`
  (`projectSlice.ts:208`). `StoryboardManager` also reads `frame_index`
  (`manager.py:372`). **Fix:** one canonical Pydantic `TimelineItem` on the write
  path, validate/normalize on `PUT`, and round-trip through the same type in
  TypeScript.
- **DATA-03 — Split shot.** `split_shot()` calls `datetime.utcnow()` with no
  `datetime` import (`projects.py:587`) → `NameError` → 500. Even if fixed, the
  client maps `s.timeline` as an array (`shotSlice.ts:135`), but the DB column is a
  JSON string, so `.map` throws; and the new shot keeps the original `index`
  (`projects.py:592`), breaking scene order. Split also inserts by `created_at`,
  so a new V1 shot appears at the end of the sequence. **Fix:** import datetime,
  serialize/parse timeline, assign a new sequence index, and shift following
  indices.
- **DATA-04 — Push to timeline.** `trimOut: shot.numFrames` makes
  `duration_frames = numFrames - trimIn - trimOut = 0`. The store still shows a
  1-frame clip because `timelineUtils` clamps to 1, masking the bug. Set
  `trimOut: 0`, persist the resulting order/start frames, and call
  `patchShot`/save instead of local-only updates.
- **DATA-05 — No migration system.** `SQLModel.metadata.create_all()` never
  alters existing tables. The repo relies on manual, git-ignored scripts
  (`add_timeline_column.py`, `migrate_v1_v2.py`, `backend/migrations/`). Any schema
  change can brick an existing studio. Adopt Alembic, checked-in migrations, and a
  startup schema-version check that refuses to start on mismatch.
- **DATA-06 — Save payload omissions.** `saveProject()` does not send
  `height`, per-shot `fps`, `shotType`, `index`, or `status`
  (`projectSlice.ts:74-103`). New shots created through autosave lose shot type
  and can inherit the model default height (512) instead of the project's. It also
  never bumps `Project.updated_at` (the line is commented out in
  `projects.py:199`), so "last modified" is meaningless.
- **DATA-07 — Destructive storyboard merge.** `commit_storyboard()` deletes any
  shot whose `(scene, index)` no longer exists (`storyboard.py:142-149`) without
  deleting or warning about its generated media. Re-parsing a shortened script can
  silently orphan finished takes. Archive removed shots or require confirmation
  when they own `video_url`/jobs.
- **DATA-08 — No referential cleanup.** Deleting an element, asset, shot or project
  does not clear references from `Shot.video_url`, `thumbnail_url`,
  `matched_elements`, or `Job.output_path`. Broken clips and dead metadata
  accumulate.
- **DATA-09 — Pending jobs survive restarts.** Startup marks only
  `status="processing"` as failed (`server.py:83`). The live DB has two stuck
  `pending` jobs. There is no persisted queue and no resume. Any job queued in
  `BackgroundTasks` dies with the process.

### Generation Orchestration

- **GEN-01 — Shot prep prompt corruption.** In
  `prepare_shot_generation`, `replacement` is assigned inside the per-match loop
  but referenced after it (`manager.py:345-346`). With an empty
  `matched_elements` list this raises `UnboundLocalError`; with matches it appends
  `. Setting: <last element>` regardless of element type, duplicating text already
  replaced. Move the setting injection inside the location branch and remove the
  unconditional append.
- **GEN-02 — Generated audio disappears from exports.** `process_video_track()`
  maps only `[i:v]`; the render filtergraph never reads `[i:a]` for V1/V2
  (`render_utils.py:72-96`). The preview also starts muted for Safari autoplay
  (`CinematicPlayer.tsx:142`) and only unmutes via a one-time global gesture, with
  no per-clip mute/gain state. So LTX's audio is absent from every exported film,
  and preview audio has no reliable control surface. **Fix:** mix V1/V2 audio into
  the render graph (with per-clip gain and mute state) and give preview a deliberate
  audio path after a user gesture.
- **GEN-03 — One semaphore, many bypasses.** `gpu_job_wrapper` wraps only
  `generate_video_task`. Thumbnails (`storyboard.py:426`), still images
  (`jobs.py:268`), element visuals (`elements.py:92`), and inpainting
  (`elements.py:161`) are dispatched directly. Two Flux calls can also run
  concurrently against the same singleton, and Flux can be loaded while LTX is
  generating. This is the fastest route to an OOM on a 32GB Mac. **Fix:** one
  scheduler for every GPU task with per-model locks and a persisted queue.
- **GEN-04 — Cancellation is not a state machine.** The video task's
  `cancelled` broadcast references `loop`, which is not defined in that scope; the
  `NameError` is swallowed (`video.py:785-790`). Chained generation checks
  cancellation before a chunk and `break`s (`chained.py:252`), then proceeds to
  stitch, probe, mark completed, and broadcast `complete`. The UI's
  `cancelled` handler therefore rarely runs. **Fix:** a single `JobCancelled`
  exception, terminal-state write in `finally`, and one `cancelled` broadcast from
  the job runner, never from task internals.
- **GEN-05 — Chained audio desync.** Video drops the first `frames_to_trim`
  frames; audio drops the last `trim_samples`. Each chunk shifts audio by ~1s.
  Trim the start of both, or better, trim in the decoder before muxing.
- **GEN-06 — Seed is always randomized for shot generation.**
  `generate_shot()` overwrites `shot.seed` with a random value before every run
  (`storyboard.py:606-613`). The inspector's seed control is therefore a lie for
  storyboard shots. Add a "lock seed / randomize" toggle; only randomize when the
  user asks.
- **GEN-07 — AI storyboard blocks the event loop with a 19B load.**
  `POST /storyboard/ai-parse` calls `manager.load_pipeline("ti2vid")` inside the
  request (`storyboard.py:217-221`), which unloads Flux via MemoryManager and runs
  Gemma synchronously. On a Mac this can freeze the whole API for minutes and
  evict the image model. Move parsing to a lightweight LLM slot (Ollama/Qwen) or a
  worker process; add a timeout and cancellation.
- **GEN-08 — SAM proxies block the event loop.** `requests.post` is called
  directly inside async handlers (`elements.py:178,205,231`; tracking in
  `tracking_manager.py` with `timeout=1200`). A long tracking propagation can
  block every other endpoint, including cancel and status. Use `httpx.AsyncClient`
  or run the synchronous call in a thread; cap propagate time and make it
  cancellable.
- **GEN-09 — Inpainting blocks and is not wired to the shot.** `process_inpaint`
  is async but calls blocking `flux_inpainter.inpaint` directly
  (`inpainting_manager.py:152`); it bypasses the GPU queue, ignores the cancel
  flag, hardcodes guidance, and its `complete` event has no `url`
  (`inpainting_manager.py:217-222`). `ServerSlice` treats it as a non-thumbnail
  and sets no `videoUrl`; the result only appears in the gallery. Run it through
  the scheduler, use `run_in_executor`, and update the originating shot/frame.
- **GEN-10 — Frame-count validation is missing.** `num_frames` from the slider
  (25–1200) is passed unvalidated. LTX expects a 1+8k grid; only mod-64 resolution
  is enforced. Reject or round invalid durations before dispatch, and surface the
  effective value to the UI.
- **GEN-11 — `auto_continue` can condition on the wrong clip.** It picks the
  latest completed job in the project (`jobs.py:181-187`), not the previous shot
  in the scene/sequence. It can extend an unrelated take.

### NLE Playback & Export

- **NLE-01 — Export timing is wrong.** See DATA-01. Render uses `start_frame`
  for V1 while the editor computes magnetic starts; with the live DB's all-zero
  starts, the last V1 clip wins and earlier clips vanish.
- **NLE-02 — V2 is not an overlay.** Each V2 clip is scaled and `pad`ed to the
  full project canvas (`render_utils.py:84-85`) then overlaid with no alpha
  (`render_utils.py:139-147`). Black padding covers V1. V2 needs
  `format=auto`/alpha-preserving compositing, or a chroma/luma key, and should not
  pad opaque black.
- **NLE-03 — Track state is ignored.** `muted`, `hidden`, and `locked` live only
  in `trackStates` and never reach the render request. Hidden V1 still renders;
  muted A1 still mixes. Send `render_settings` with track flags or filter on the
  backend.
- **NLE-04 — Audio mix truncation.** `amix=duration=first`
  (`render_utils.py:157`) ends the mix at the first input's duration. A short
  first clip silences later music/SFX. Use `duration=longest` plus explicit
  `-t`/`-shortest` policy, and add per-clip gain/fades.
- **NLE-05 — Audio duration ignores `trimOut`.** `GlobalAudioManager` computes
  `duration = numFrames / fps` (`GlobalAudioManager.ts:220`), not
  `(numFrames - trimIn - trimOut) / fps`, so trimmed audio keeps playing past the
  visual cut and can overlap the next clip.
- **NLE-06 — No transitions or effects.** There is no crossfade, fade-to-black,
  speed change, LUT, title, or subtitle path in the render graph. This is a
  product gap, but also a correctness gap for any cut that expects a transition.
- **NLE-07 — Render is not cancellable or resumable.** Render jobs are not
  registered in `active_jobs`, so `POST /jobs/{id}/cancel` cannot stop them. A
  crashed render leaves `processing` and is only cleaned on next startup.
- **NLE-08 — Render progress is fake-granular.** Progress jumps 5 → 10 → 30 →
  100; FFmpeg's `-progress` output is not parsed. Long exports look stalled.

### Jobs, SSE & Cancellation

- **JOB-01 — No durable queue.** Jobs are `BackgroundTasks` plus an in-memory
  `active_jobs` dict. They do not survive restart and cannot be reordered.
  `_gpu_queue_count` counts waiters but is not a real queue.
- **JOB-02 — Pending jobs leak.** See DATA-09.
- **JOB-03 — `cancelled` is not reliably emitted.** See GEN-04. `jobPoller` also
  has no `cancelled` branch (`jobPoller.ts:31-50`), so a refresh during/after
  cancellation can leave `isGenerating=true`.
- **JOB-04 — No SSE replay or reconnect sync.** `SSEProvider` reconnects on
  `onerror` but does not re-run `pollJobStatus` for in-flight shots
  (`SSEProvider.tsx:49-60`). A completion during the outage is lost until a full
  page reload. Add an event ID/ring buffer or, simpler, poll all active jobs on
  every reconnect and on `visibilitychange`.
- **JOB-05 — Stale connection detection is incomplete.** `lastEventTime` is
  stored but never used; the browser may not fire `onerror` promptly. Add a
  heartbeat timeout (e.g., 40s without any event → close/reconnect).
- **JOB-06 — Duplicate route registrations.**
  `PATCH /shots/{shot_id}` exists in both `routes/projects.py:334` and
  `routes/shots.py:12`; the first wins, so the JSON-normalizing implementation in
  `shots.py` is dead. `GET /models/disk` is registered twice
  (`routes/models.py:75, 524`) with different response shapes. Remove the dead
  handlers and add a route-uniqueness test.
- **JOB-07 — No idempotency or per-shot lock.** Double-clicking Generate can
  create two jobs for the same shot; the second overwrites `lastJobId` and the
  first becomes an orphan whose completion may attach to the wrong state. Reject a
  second active job for the same shot or make job creation idempotent.

### Frontend State & Contracts

- **FE-01 — Hardcoded localhost.** 74 occurrences of
  `http://localhost:8000` across 20 files, while only `ModelSettings` reads
  `VITE_API_URL`. Opening the studio from another machine or a Tauri sidecar
  breaks REST, SSE, media, audio, and polling. Centralize an API client and asset
  resolver; derive from `window.location` when unset.
- **FE-02 — Contract drift is structural.** `Shot` (camelCase) ↔ `Shot` DB
  (snake_case) ↔ `ProjectState.shots: List[dict]` (untyped) means TypeScript
  cannot catch the `frameIndex`/`frame_index`, `height`, or `shotType` losses.
  Generate types from OpenAPI or define shared Pydantic/TS contracts and validate
  at the boundary.
- **FE-03 — Optimistic updates never roll back.** `patchShot` applies locally,
  then ignores non-2xx responses (`shotSlice.ts:87-96`). A failed save looks
  successful until reload. Keep a previous snapshot, roll back on error, and show
  a save state.
- **FE-04 — Autosave races.** `useAutoSave` PUTs the whole project 2s after any
  change (`VisualTimeline.tsx:10-33`). Two overlapping PUTs can commit out of
  order; there is no version/ETag check. Add a monotonic revision or a serialized
  save queue, and skip saves when nothing changed.
- **FE-05 — Undo history is polluted.** zundo tracks the whole `project` object,
  including `progress`, `isGenerating`, `statusMessage`, and prompt streams. Every
  progress event creates a history entry, so 20 undo steps can cover only a few
  seconds of generation. Track editing fields only, or pause temporal during
  server-driven updates.
- **FE-06 — `isGenerating` persists across reload.** Persist partializes the whole
  project; `merge` resets toasts/playback/editing but not per-shot transient
  state. If the job died, the UI shows a permanent spinner until poll returns 404
  (which `jobPoller` ignores). Reset transient fields on hydrate and treat 404 as
  "not running".
- **FE-07 — Storyboard edits can be lost.** `commitEdit` debounces the PATCH by
  300ms (`StoryboardShotCard.tsx:60-68`); unmounting or switching scenes before
  the timer fires drops the edit. Flush on blur/unmount.
- **FE-08 — Drag preview uses undefined fields.** `TimelineClip` reads
  `clip.numFrames` and `clip.name` (`TimelineClip.tsx:31,103`), but the computed
  clip exposes `duration` and `shot`; the transient duration becomes `NaN`.
- **FE-09 — Per-frame store writes still notify broad subscribers.** The headless
  subscribers are designed for this, but `Layout`, `VisualTimeline` and
  `CinematicPlayer` select `project`/`shots`, so every 60fps `setCurrentTime`
  reconciles large trees. Split playback state from document state or use
  `subscribeWithSelector` + narrow keys.
- **FE-10 — Deletion is local-only in two paths.** `Layout`'s Delete key calls
  local `deleteShot` without a DELETE request; it relies on the next autosave to
  prune the DB. If the tab closes first, the row survives. Use the storyboard
  DELETE endpoint or a tombstone in save.

### Model Management

- **MOD-01 — Activation is a no-op.** `POST /models/{id}/activate` calls
  `memory_manager.prepare_for(slot)` and sets status `active`
  (`routes/models.py:292-312`). `prepare_for` only unloads conflicts; it does not
  load. `model_loader.activate_base_model()` and `deactivate_base_model()` exist
  but are never called. The UI reports success while no model is loaded.
- **MOD-02 — Disk report points at the wrong directories.**
  `model_settings._get_model_dirs()` uses `<root>/ltx2/models` and
  `<root>/sam3/checkpoints` (`model_settings.py:86-92`); the actual paths are
  `LTX-2/models` and `backend/models/sam3`. The dashboard under-reports.
- **MOD-03 — Duplicate `/models/disk`.** First handler returns registry byte
  totals; second returns `model_settings` breakdown. The first wins, so the
  richer shape is unreachable. Consolidate into one schema.
- **MOD-04 — Downloader bypass and stale API usage.** `POST /models/{id}/download`
  calls `download_model` directly (`routes/models.py:153-154`), bypassing the
  priority queue; concurrent downloads can hit the same disk. The downloader uses
  `local_dir_use_symlinks` (`model_downloader.py:369,409`), which is deprecated in
  current `huggingface_hub`, and looks for a `.downloading` suffix
  (`model_registry.py:163`) while HF uses `.incomplete`. Cancel of a queued (not
  active) download returns 404, which the UI treats as failure.
- **MOD-05 — LoRA reconstruction is fragile.** `ModelManager.load_pipeline`
  caches by `pipeline_type` only (`model_engine.py:68-70`); comments admit LoRA
  changes are ignored. `model_loader` reconstructs by deleting the pipeline
  outside the manager's lock (`model_loader.py:280-292`), so a generation in
  flight can lose its pipeline. The weird walrus removal in `remove_lora`
  (`model_loader.py:184-195`) is dead code that happens to work.
- **MOD-06 — MPS dtype claim is not in the loader.** README/docs say the
  transformer is forced to `float32` on MPS; `model_engine.py` only casts the VAE
  (`:145-148`) and passes `fp8transformer=False`. Either restore the cast or
  correct the docs and prove bf16 stability on the supported PyTorch version.

### Ops, Testing & Packaging

- **OPS-01 — No tests.** `tests/` is empty; no `pytest`, `vitest`, or Playwright
  setup. There is no golden test for the two most fragile things: save/load
  round-trip and timeline render output.
- **OPS-02 — No CI.** No workflow file, no lint/type/test gate. `npm run lint`
  fails today; `compileall` cannot catch the runtime `NameError` in split.
- **OPS-03 — Dependency drift.** `backend/requirements.txt` omits `requests` and
  `huggingface_hub` (both imported), pins `torch` but not `transformers` (while
  `flux2` pins `transformers==4.56.1`), and has no dev/test extras or lockfile.
- **OPS-04 — The documented environment is broken in this checkout.** The
  `milimov` venv's editable `.pth` files point at
  `/Users/mck/Desktop/milimovideo/...`, not this repo, so `./run_backend.sh` fails
  with `ModuleNotFoundError: ltx_pipelines`. Add a `verify_setup` command that
  checks imports, FFmpeg, model paths, and writable directories before startup.
- **OPS-05 — No LICENSE file.** README claims Apache 2.0 and the badge links a
  `LICENSE` that does not exist. Add `LICENSE` and a `NOTICE` covering the LTX-2,
  FLUX.2 and SAM 3 licenses, plus the model-license gates.
- **OPS-06 — Debug artifacts on the hot path.** Every video job writes
  `server_last_request.json` and failures write `server_last_error.txt` in the
  process CWD (`video.py:702,799`). These contain prompts and absolute paths, and
  race when jobs overlap. Gate behind a debug flag and rotate.
- **OPS-07 — Logging is not request-correlated.** The rotating file handler uses
  a relative `server.log`, so the location depends on CWD; there is no request ID,
  job ID context, or structured exception field beyond the traceback string.

## What Already Works

Be fair to the build — these are not small things:

- **Pipeline discipline is real.** One LTX pipeline cached at a time, explicit
  unload → `gc.collect()` → device cache flush, and Flux's AE hot-swap are solid
  foundations (`model_engine.py:59-151`, `flux_wrapper.py:639-659`).
- **MemoryManager's intent is correct.** Mutual exclusion between video and image
  is the right architecture; it needs to become the *only* way to touch a GPU,
  not advice that four routes ignore.
- **SSE transport has backpressure and TTL.** Per-client bounded queues, stale
  message drop, keepalives, and reconnect backoff are better than most local apps
  (`events.py:16-93`, `SSEProvider.tsx:33-153`).
- **Prompt enhancement is thoughtfully layered.** Gemma vs Ollama dispatch,
  vision-model detection, reference-sheet fallback, and `keep_alive: 0` unload
  discipline are good product engineering (`llm.py`).
- **The storyboard engine is the durable IP.** Regex parsing, element matching
  with provenance, concept-art thumbnails, and shot preparation are a coherent
  workflow; they need correctness fixes, not a redesign.
- **SAM 3 isolation holds.** Port 8001 as a separate service is still the right
  call; the proxy surface is small and replaceable.
- **Safari media serving exists for a reason.** Range requests are implemented;
  they just need safe path handling and streaming.
- **Asset/project workspace layout is sane.** `projects/{id}/{generated,
  thumbnails,assets,workspace}` is a good local-first shape.

## Missing Functionality (Product Gaps)

| Capability | Status | Why it matters |
|---|---|---|
| Take/version history per shot | Missing | Regeneration replaces the shot's URL; old takes are only findable in the global gallery, with no compare/rollback |
| Persistent/resumable job queue | Missing | Closing the app or a crash loses queued and in-flight work |
| Render presets (codec, bitrate, resolution, fps, audio normalize) | Missing | One hardcoded H.264 CRF 18 path; no delivery control |
| Transitions, fades, speed, LUT, titles/subtitles | Missing | Cannot finish a real cut inside the studio |
| Audio from V1/V2 clips, gain, fades, ducking | Missing/Broken | Generated audio is inaudible and unexported |
| Project bundle import/export, media relink, proxies | Missing | Projects are not portable; large media kills playback |
| Real model activation + warmup + OOM fallback | Missing/Broken | "Active" is cosmetic; first generation pays the load cost and OOM has no recovery |
| Seed lock / deterministic re-run | Broken | Storyboard generation always randomizes |
| API auth / origin policy | Missing | Local-first does not mean open-to-any-webpage |
| Tests, CI, migration gate | Missing | Regressions are found by users, in the editor |
| Eval harness (VBench subset + blind A/B) | Missing | v2 model swaps cannot be justified |
| Character LoRA training, 3D previz, agent copilot | Roadmap only | Correctly deferred; do not start before the foundation is stable |

## Phase 0 Verification — Doc Contradictions Resolved

The v2 roadmap's Phase 0 asked for the code truth. Here it is:

| Claim in docs | Code truth |
|---|---|
| "Five routers" | **Seven**: projects, jobs, assets, storyboard, elements, shots, models |
| React 18 | **React 19.2** (`web-app/package.json`) |
| Chained generation triggers above 505 frames | Triggers above `config.NATIVE_MAX_FRAMES = 121`; chunk size 121, overlap 25 |
| `NATIVE_MAX_FRAMES = 121` comment says "~20.2s at 25fps" | 121 / 25 = **4.84s**; 20.2s is the old 505-frame number. The comment is stale |
| `cfg_scale` default 2.0/3.0 | DB `Shot.cfg_scale = 1.0`, `ShotConfig = 2.0`, frontend `DEFAULT_SHOT = 3.0` |
| `upscale` default true/false | DB `false`, `ShotConfig = true`, frontend `true`, task fallback `false` |
| Endpoint `POST /generate_advanced` | Actual `POST /generate/advanced`; no underscore alias |
| Save endpoint `PUT /projects/{id}/save` | Actual `PUT /projects/{id}` |
| `/track/remove` unclear | Backend exposes `POST /track/remove`; proxies SAM's `/track/remove_object` |
| Inpaint SSE/Asset detail undocumented | `complete` with `type:"inpaint"` and `asset_id` exists, but **no `url`**, and the store does not update the shot |
| Transformer forced to float32 on MPS | Not in `model_engine.py`; only VAE is cast |
| Chained uses latent handoff / `-c copy` | Current code uses 25-frame pixel `media_sequence`, trims in Python, re-encodes with `concat` filter |
| "No optimistic updates" | `patchShot` is optimistic and never rolls back |
| `ElementSlice` owns `pushStoryboardToTimeline` | It lives in `ShotSlice` |

## Related

- [[../roadmap/implementation-plan-2026-11.md]] — the sequenced fix plan
- [[../roadmap/frontend-refactor-2026-11.md]] — the studio shell rebuild
- [[studio-shell.md]] — target front-end architecture
- [[../research/frontend-video-platforms-2026.md]] — platform/canvas research
- [[../roadmap/v2-roadmap.md]] — model/architecture roadmap, now gated behind this work
- [[architecture.md]] — system map and router inventory
- [[data-models.md]] — persistence layer
- [[nle-timeline.md]] — timeline semantics
- [[sse-eventing.md]] — event transport and cancellation
- [[../pipelines/chained-generation.md]] — chained generation behavior
- [[../pipelines/inpainting.md]] — inpaint path
