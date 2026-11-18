# Milimo Video Wiki Log

Append-only chronological record. Every entry starts with a parseable prefix:
`## [YYYY-MM-DD] action | Title`.

## [2026-10-09] init | Wiki created from llm-wiki pattern

Built the wiki following the llm-wiki pattern (Karpathy gist): raw sources stay
immutable under `docs/` and `dev_diary/`, the LLM owns `wiki/`, and `AGENTS.md`
is the schema. Seeded from `README.md`, `docs/01_system_architecture.md`,
`docs/03_ai_pipelines.md`, and `docs/user_guide.md`. Pages created: overview,
architecture, nle-timeline, ltx-2, flux-2, sam-3, chained-generation,
memory-slots, mps-first, getting-started, sam3-microservice decision.

Not yet ingested: `docs/02_data_models.md`, `docs/04_frontend_state.md`,
`docs/05_execution_flow.md`, `docs/06_file_dependency.md`, `docs/flux2-bible.md`,
`docs/ltx2-bible.md`, `docs/sam3-bible.md`, `docs/CODEBASE_ANALYSIS.md`, and the
`dev_diary/` build log. Natural next ingests: a Gemma 3 / prompt-enhancement
models page, a storyboard-engine systems page, and an SSE/eventing concepts page.

## [2026-10-09] ingest | docs/02_data_models.md + docs/06_file_dependency.md + docs/CODEBASE_ANALYSIS.md

New page `systems/data-models.md` (SQLite/SQLModel entities, Pydantic schemas,
TypeScript types, storage layout). Enriched `systems/architecture.md` with a
Module Map section (backend/frontend/SAM-3 layout, key import edges, 5-router
confirmation). Contradictions flagged, not resolved: cfg_scale default 3.0 (ER)
vs 2.0 (ShotConfig); upscale default false (entity) vs True (ShotConfig);
endpoint naming `/generate_advanced` vs `/generate/advanced` (and
`/shots/{id}/generate` in the frontend docs); backend file count 24 vs 28;
`model_engine <-> memory_manager` cycle vs the "no circular dependencies" claim.

## [2026-10-09] ingest | docs/04_frontend_state.md + milimo-frontend-state-manager skill

New page `systems/frontend-state.md` (7-slice God Store inventory, persist/zundo
config, selector discipline, ServerSlice event routing, SKILL.md conventions:
no new contexts, magnetic V1 layout rules, Safari drag-hover workaround). No
contradictions with existing pages.

## [2026-10-09] ingest | docs/05_execution_flow.md

New pages `systems/sse-eventing.md` (EventManager, /events, SSEProvider backoff,
jobPoller recovery, event payload shapes incl. is_thumbnail matching) and
`systems/storyboard-engine.md` (regex vs Gemma parsing, ElementMatcher,
commit, thumbnails, shot generation prep); `pipelines/inpainting.md` gained its
mask-acquisition paths and Job persistence. Note: `log` SSE event documented in
01 but has no payload examples in 05.

## [2026-10-09] ingest | docs/flux2-bible.md

Enriched `models/flux-2.md` with Internals & Gotchas (init sequence, denoise-loop
mechanics, IP-Adapter 4-token injection stripped in the uncond pass, AE hot-swap
mechanics, inpaint call signature, config keys). New page `pipelines/inpainting.md`
(SAM mask -> Flux RePaint noisy-known blend, job persistence, failure modes).
Flagged: bible lists two 9B variants without saying which `flux-2-klein-9b.safetensors`
is; the bible documents no thumbnail/Asset/SSE detail for the inpaint path itself.

## [2026-10-09] ingest | docs/ltx2-bible.md

Enriched `models/ltx-2.md` with Internals & Gotchas (rectified-flow 50->8 step
distillation, constructor args, IC-LoRA skipping distillation, keyframe 2-image
contract, indexed conditioning tuples, guidance 1.0, URL resolution, pipeline
type-string cache gotcha, MPS diagnostics). Enriched
`pipelines/chained-generation.md` with Refinements ("Smart Continue" name,
ti2vid-AND->505 precondition, latent tail-slice handoff, -sseof -0.1, -c copy
trim template, per-chunk SSE). Note: bible omits the concept-art-thumbnail
fallback in the conditioning priority chain.

## [2026-10-09] ingest | docs/sam3-bible.md

Enriched `models/sam-3.md` with Internals & Gotchas (startup sequence, stateful
Sam3Processor, point labels, multimask, 1008px resize, temporal disambiguation
tracker, bidirectional propagation, max_frames=-1 sentinel, backend proxy routes,
save/load format, MPS port catalog). New page `workflows/masking-tracking.md`
(MaskingCanvas 4 modes, TrackingPanel lifecycle, mutual exclusivity, export).
Flagged: `01_system_architecture.md` lists `POST /track/remove` but the bible
documents only start/prompt/propagate/stop — verify against code.

## [2026-10-09] ingest | dev_diary/

New page `concepts/project-history.md`: distilled build timeline (Jan-Feb 2026),
key milestones (Quantum Latent Alignment, SAM 3 port, storyboard+elements,
Flux inpainting, MPS campaigns, SQLite/modular refactor, 2026-02-17 acquisition
announcement with open-source continuation as stated non-negotiable term), and
open threads (cross-scene consistency, camera controls, vision prompting). Facts
distilled in own words; no marketing copy pasted.

## [2026-10-09] ingest | mv_plan/google_cloud_integration.md

New page `decisions/cloud-engines.md` (PROPOSAL, not implemented): optional
per-engine cloud fallbacks (Veo 3.1 / Imagen / Gemini via Vertex AI) behind a
CloudSettings UI, bypassing `gpu_job_wrapper` so local-first stays the default.
`mv_plan/script_example.md` ("NEON BLOODLINE" demo screenplay) skipped as demo
content, not engineering knowledge. `skills/` surveyed: only
`milimo-frontend-state-manager/SKILL.md` was project-relevant (ingested); the rest
are generic skill packs.

## [2026-10-09] research | Oct 2026 landscape survey (5 domains)

Five parallel web-research workers surveyed the state of the art for the v2
planning effort; all post-Jan-2026 claims are web-sourced, unconfirmed items
marked in-page. New pages under `wiki/research/`: `video-models-2026.md`
(LTX-2.5 / Wan 2.2+3.0 / MiniMax H3 / Kandinsky 6.0 / HunyuanVideo-1.5;
licensing traps documented), `image-models-identity-2026.md` (Z-Image Turbo /
FLUX.2 / Qwen-Image line / native multi-ref conditioning / LoRA identity),
`world-models-3d-camera-2026.md` (world models / TRELLIS.2 / Depth Anything 3 /
keep-SAM-3 verdict / ProPainter / Netflix VOID),
`inference-infra-2026.md` (vLLM-Omni / SGLang-diffusion / MLX DiT ports /
PyTorch MPS 2.13-2.14 / FP8-NVFP4-GGUF quantization; slot architecture verdict:
keep), `frontend-ux-workflows-2026.md` (React 19 / canvas timelines / Tauri 2 /
copilot UX / Google Co-Director agentic frameworks / Qwen3-VL 8B / VBench-2.0).

## [2026-10-09] roadmap | Milimo Video v2 design roadmap

New page `wiki/roadmap/v2-roadmap.md`: guiding principles (local-first kept, slots
kept, MPS-first relaxed on float32), per-slot recommendations (video: LTX-2.5 +
Wan 2.2; image: Z-Image Turbo + Qwen-Image-Edit-2511 + per-character LoRA; keep
SAM 3; LLM: Qwen3.x + Qwen3-VL 8B), architecture deltas (SGLang spike, MLX spike,
FP8 slots, GGUF distribution), UI/UX direction (React 19, canvas timeline, Tauri
2 sidecar, copilot panel, LangGraph supervisor crew, MCP server), phased plan
(Phase 0 verify-against-code → Phase 1 eval+models → Phase 2 architecture →
Phase 3 UX), risks, and 7 open questions for Mainza. Research only; no
implementation. AGENTS.md amended: new `research` and `roadmap` page types with
section conventions, `⚠️ VERIFY` flag convention for doc-vs-code claims.

## [2026-11-16] audit | Deep-dive engineering audit + implementation plan

Audited the current `main` (807985e): all first-party backend modules, the
frontend store and generation/playback/export components, the FastAPI route table,
the live SQLite DB, `compileall`, `tsc`, and `npm run lint`. New pages:
`systems/engineering-audit-2026-11.md` (severity-ranked P0/P1 findings with
file:line evidence, what already works, missing functionality, resolved doc
contradictions) and `roadmap/implementation-plan-2026-11.md` (6-week stabilization
plan: security hotfixes → canonical timeline → render fidelity → unified GPU queue
→ model-management truth → frontend contracts/CI, then the gated v2 model lane).
Key findings: `eval()` RCE in `/edit/inpaint`; unsanitized upload filenames;
open CORS/no auth; V1 order never persisted and all-zero `start_frame` makes export
overlay every V1 clip at t=0; `frameIndex`/`frame_index` mismatch loses
conditioning; split shot is broken by a missing `datetime` import plus a client
`.map` on a JSON string; push-to-timeline creates zero-length clips; export drops
V1/V2 audio and V2 overlays black; image/inpaint jobs bypass the GPU queue; video
cancellation never emits `cancelled`; model "activate" is a no-op; no tests/CI/
LICENSE. Updated `roadmap/v2-roadmap.md` (Phase 0 complete, Phase 1 gated by
G2/G3), `index.md`, and the affected system pages with audit corrections. No code
changed by this audit.

## [2026-11-16] research | Frontier video/canvas platforms + Kubona frontend

Surveyed 20 frontier video, storyboard, editing, review, and canvas platforms
(Runway Gen-4/Aleph/Agent, Google Flow/Veo 3.1, Kling 3.0, Luma Agents, Pika,
LTX Studio, Adobe Premiere/Firefly + Frame.io, Descript, CapCut, OpusClip, Boords,
Shot Designer/StudioBinder, Figma Weave, Krea, ComfyUI, tldraw, Milanote, Sora 2
as a discontinued interaction reference). Cloned and inspected Kubona
(`platform/console`) at `6d1e2cb`: Workbench spatial canvas (camera, zoom/LOD,
viewport culling, areas, typed links, pins, marquee, align/distribute, minimap,
follow-live, list view, conflict-aware layout saves), ModeRail/CommandDock/
Inspector/CopilotDrawer/CommandPalette shell, and the lazy/fail-closed 3D
observatory. New page `research/frontend-video-platforms-2026.md` (profiles,
comparison matrix, 15 patterns, anti-patterns, relevance to Milimo). Key synthesis:
the winning loop is conversation → board → cut over one project graph;
consistency is an asset problem (Ingredients/Elements/Master References); takes
must be immutable; agents must plan before spending; local-first is Milimo's
unclaimed moat.

## [2026-11-16] roadmap | Studio shell redesign + frontend refactor plan

New pages `systems/studio-shell.md` (target architecture: Story/Generate/Edit/
Review/Library/Deliver surfaces, canonical Project→Sequence→Scene→Shot→Take graph,
board/timeline as peers, four stores, canvas engine with LOD/culling/minimap/list,
timeline engine with V1 order + V2/A1/A2/captions, command dock/agent with
plan-before-spend, contracts, performance budgets, accessibility) and
`roadmap/frontend-refactor-2026-11.md` (strangler `/studio` migration, F0–F5,
gates, 12–14 weeks with two FE engineers, risks, decisions). Updated
`roadmap/implementation-plan-2026-11.md` (WS10 + front-end program, staffing,
risks, definition of done), `roadmap/v2-roadmap.md` (UI/UX direction now concrete,
Phase 3 links the shell), `systems/frontend-state.md` and `systems/nle-timeline.md`
(current vs target), `index.md`. Design stance: board for meaning, timeline for
time, one shot graph; no canvas-position-as-order; no destructive regeneration;
no God store; no hardcoded localhost.

## [2026-11-16] implementation | Execute implementation-plan-2026-11 (backend gates)

Executed the backend correctness program on `fix/implementation-plan-2026-11`.
P0: removed `eval` RCE, sanitized/limited uploads, loopback CORS + optional token,
safe streaming media ranges, fixed split/push/manager prompt injection, unified
cancellation broadcasts (video/image/render; chained raises instead of stitching),
startup recovery for pending/processing/queued, removed duplicate routes, fixed
chained audio trim, added LICENSE/NOTICE. P1: `Shot.sequence_index` + backfill,
`Project.revision` optimistic concurrency, canonical `TimelineItem`, full save
payload, central API client/no hardcoded localhost, serialized saves, undo
equality excludes job progress, transient hydrate reset, 404 poll recovery.
P2: render from computed V1 layout, V2 alpha compositing, V1/V2 embedded audio,
track states, `amix duration=longest`, FFmpeg progress/cancel, render presets;
all GPU work now passes a durable `QueuedJob` ledger + one global semaphore,
with startup resume/one retry and idempotent per-shot jobs. P3: real model
activate/deactivate, corrected disk paths, queued/resumable download handling,
requirements, `verify_setup.py`, debug artifact gating, VRAM preflight. P5:
pytest suite (21 passing), vitest store/layout + shell tests (4 passing), CI workflow,
pre-commit, `/health` device/queue/slots/disk. Verified: live server health, real
FFmpeg render (4.0s video+audio), tsc, eslint 0 errors, vite build. Deferred: Alembic baseline, durable JobQueue,
takes/version history, vitest/Playwright, and the F0–F5 studio shell refactor
(single-engineer guidance: backend gates first). Updated
`roadmap/implementation-plan-2026-11.md` and `systems/engineering-audit-2026-11.md`
with status.

## [2026-11-16] frontend | F0 studio shell + F1 board prototype

Added the strangler `/studio` shell (lazy route; legacy editor preserved at `/`)
and server-persisted board layout/view endpoints (`/projects/{id}/board`,
`/projects/{id}/view`) with revision/CAS:
design tokens/graphite+cyan CSS, mode rail, top bar, inspector, context-aware
command dock with plan preview, status strip, persisted editor store. Story
surface: scene/shot spine, infinite board with pan/zoom/card drag/LOD/minimap,
accessible shot-list view, script view, selection sync. Edit surface: stage
monitor, legacy playback clock, V1/V2/A1 timeline shell, playhead, seeking,
zoom. Generate/Review/Library/Deliver surfaces are real data-backed shells over
existing endpoints, with explicit F3/F4 gaps rather than fake behavior. Added
immutable take history/select via the job ledger (`/shots/{id}/takes`,
`/shots/{id}/select-take`) and surfaced it in Generate/Inspector/Review. Added
happy-dom + Testing Library mount test for the shell. Verified: tsc, eslint 0
errors, 4 vitest tests, vite build (StudioApp chunk 58 kB). Updated the
frontend-refactor and implementation-plan status tables.

## [2026-11-16] frontend | F1 areas and typed links

Completed the F1 board semantics: draggable/resizable areas with inline titles and
palette colors; typed links (thread, blocking, reference) drawn as bezier curves
with deletable midpoint handles; link-source card highlight. Both are persisted
through the existing `/projects/{id}/board` revision/CAS payload, so layout,
areas, and links save atomically while camera stays in the separate ephemeral view
record. Added editor-store unit tests for area CRUD and typed-link dedupe.
Verified: 21 pytest, 6 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] agent | local plan API and master-reference filter

The command dock no longer fakes plans. `POST /projects/{id}/plan` is a pure,
local, deterministic planner: explicit directives (prompt:, negative:, seed,
frames, shot type, add/append) become diff steps with before/after, plus
shot-type alias normalization, rough GPU/VRAM estimates for generate plans, and
honest warnings when nothing matches. Ask returns project/shot facts; Find
returns a board filter. The client applies approved patches through the
undoable document store, and generation only runs after approval. Added the
master-reference chip list and filter to the story spine, moved the board
filter into the editor store so Find can drive it. Verified: 25 pytest,
6 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] frontend | frame-accurate review comments and approvals

Added the review persistence contract: `review_item` stores project comments
(shot + frame + body + status) and a single upserted approval per project/shot
(`approved` | `changes_requested`). Endpoints: `GET /projects/{id}/review`,
`POST/PATCH/DELETE .../review/comments`, `POST .../review/approvals`. The Review
surface now reports the live frame from the player, attaches comments at that
frame, jumps to a comment's frame/shot, resolves/deletes comments, and records
shot decisions; the version stack and approval badges read the same data. Added
3 backend tests and a review-surface mount test. Verified: 28 pytest, 7 vitest,
tsc clean, eslint 0 errors, vite build. Live DB backed up and `review_item`
created.

## [2026-11-16] frontend | frame-accurate timeline operations

Added pure timeline ops with unit tests: trim in/out within source bounds, slip
(source window moves, duration constant), roll (cut moves between adjacent V1
clips), and blade-frame calculation. StudioTimeline now has Select/Slip/Roll
tools, trim handles, blade at playhead via the existing split endpoint, ripple
delete, and gesture previews that commit once on pointer-up so undo stays
useful. V1 remains magnetic; ripple reflow comes from computeTimelineLayout.
Verified: 28 pytest, 11 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] audio | per-clip gain and mute through the render path

Added `shot.audio_gain` and `shot.audio_muted` (idempotent migration), serialized
them with the project, mapped them through the frontend document store, and used
them when building render clips instead of hardcoded 1.0/False. The studio
Inspector audio tab now has a 0–200% gain slider, mute/reset, and trim/track
readouts. Added filtergraph tests proving gain is emitted, muted clips keep video
but drop audio, and hidden/muted track states win. Verified: 32 pytest, 11 vitest,
tsc clean, eslint 0 errors, vite build. Live DB backed up and columns added.

## [2026-11-16] review | @Agent comment handoff

Comments that mention @Agent now expose "Plan fix in command dock": the editor
selects the comment's shot, switches the dock to Direct, and loads the note
(minus the mention) as a draft. The existing deterministic planner still owns
the plan/approval boundary, so an agent note cannot spend GPU time on its own.
Verified: 32 pytest, 11 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] captions | C1 cues and burned-in ASS

Captions are now first-class project data: sanitized cues persist on `project`
(`start_frame`, `end_frame`, `text`), the timeline has a C1 lane with an in-place
cue editor and "add at playhead", and the render writes a styled ASS file and
applies `ass=` after V1/V2 compositing, then cleans up the temp file. Deliver
shows the cue count and gained 1:1 and 4:5 presets alongside 16:9/9:16. Added
caption unit tests plus an end-to-end render assertion with a real cue. Verified:
35 pytest, 14 vitest, tsc clean, eslint 0 errors, vite build. Live DB backed up
and `project.captions` added.

## [2026-11-16] shell | undo/redo for approved plans

Surfaced the document temporal store in the studio: top-bar undo/redo buttons
with disabled states and ⌘Z / ⇧⌘Z shortcuts that ignore text fields. Approved
plan patches remain a single document step, satisfying the "one undo restores
pre-plan state" contract; added a store test that applies a plan-shaped patch and
undoes it. Verified: 35 pytest, 15 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] board | viewport culling for large stories

Only cards intersecting the viewport (plus overscan) are mounted, with the
visible/total count shown in the zoom pill. Areas, links, minimap, and selection
still read the full layout, so culling changes cost, not semantics. This is the
first half of the F1 500-card gate; a real 500-card perf measurement is still
outstanding. Verified: 35 pytest, 15 vitest, tsc clean, eslint 0 errors, build.

## [2026-11-16] library | project-wide take lineage

Added `GET /projects/{id}/takes` returning every completed output with shot
label, sequence index, seed/model/frames, selection state, and timestamp. The
Library gained a Takes tab with local filtering and select-for-shot, so lineage
is visible across the project instead of one shot at a time. Verified: 36 pytest,
15 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] setup | real device, pipeline and model status

Replaced the dead Setup rail button with a Setup surface backed by `/health`,
`/models`, and `/models/pipelines`: device, active slots, queue, auth mode, disk,
per-pipeline readiness with missing counts, and a base-model table showing real
status, size, and VRAM estimate. Activation/unload call the real endpoints and
are explicit, single-slot operations. Added a mount test that reaches Setup.
Verified: 36 pytest, 16 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] review | needs-you tray

The review hook now refreshes every 15s, and the top bar shows a "needs you"
pill (open comments + change requests) that jumps to Review. This is the
editor-side half of F4-06; the reviewer share link remains open.
Verified: 36 pytest, 16 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] review | capability share links

Added `share_link` (token hash only) plus `POST /projects/{id}/share`,
`DELETE /projects/{id}/share/{id}`, and `/share/{token}` read + comment/approval
routes. Share routes are read-only over the document; the reviewer page at
`/studio/share/<token>` plays the cut, attaches comments at a frame, and records
approve/changes decisions. Links are revocable and disabled when
`MILIMO_API_TOKEN` is set (media would otherwise need the studio token). Top-bar
Share copies the link. Added backend lifecycle/read-only tests and a shared-page
mount test. Verified: 39 pytest, 17 vitest, tsc clean, eslint 0 errors, build.
Live DB backed up and `sharelink` created.

## [2026-11-16] audio | place uploaded audio on A1

Audio-only shots now work end to end: the timeline Audio picker lists real
uploaded audio, reads duration from metadata, and adds an A1 clip at the
playhead; the render path already handled standalone A1 and now is covered by a
real FFmpeg render test with an audio-only input. Generate excludes A1 clips from
the shot queue so audio cannot be accidentally sent to a video pipeline.
Verified: 39 pytest, 17 vitest, tsc clean, eslint 0 errors, vite build.

## [2026-11-16] motion | opacity keyframes through the render

Added `shot.keyframes` JSON (`frame`, `opacity`), sanitized server-side,
serialized with the project, and mapped through the document store. The render
builds a piecewise-linear expression and multiplies the clip alpha with `geq`
before PTS shift, so keyframes stay clip-local; V2 fades over V1 and V1 fades
through black. Inspector gained a Motion tab to add/remove keyframes at the
playhead, scrub opacity, and clear the curve. Covered by expression/filtergraph
tests and a real FFmpeg render with keyframes. Verified: 41 pytest, 20 vitest,
tsc clean, eslint 0 errors, build. Live DB backed up and column added.

## [2026-11-16] board | extracted and tested culling

Moved viewport culling into a pure `cullCards` helper and covered it with a
500-card test (a small slice mounts at detail zoom, all stay when far out,
overscan keeps edge cards). This replaces the inline predicate; behavior is
unchanged. Verified: 41 pytest, 23 vitest, tsc clean, eslint 0 errors, build.

## [2026-11-16] ci | Playwright smoke

Added Playwright with a headless Chromium smoke test that stubs the API and
walks the real `/studio` shell: story spine, board Area control, master
references, Review frame-comment surface, and Setup memory/model panels. Wired
`npm run test:e2e`, excluded `e2e/` from Vitest, ignored Playwright artifacts,
and added the browser-install + e2e steps to CI. Also extended the real render
test with a V2 overlay carrying opacity keyframes. Verified locally: 1
Playwright pass, 41 pytest, 23 vitest, tsc clean, build.

## [2026-11-16] status | front-end plan table refreshed

Updated the implementation-plan status table to match the landed studio work:
F0/F1 mostly done, F2 mostly done (render-verified), F3/F4 partial, F5 still
deferred, P4 evidence expanded. Remaining named gaps: activity feed, full CRUD
Playwright flow, Alembic baseline, OpenAPI-generated types, P3 OOM retry.

## [2026-11-16] review | merged activity feed

Review now shows latest activity across comments, decisions, and generated takes
with author/frame/model context, derived from the existing review and take
contracts. This closes the named F4 activity-feed gap. Verified: 23 vitest,
tsc clean, eslint 0 errors.

## [2026-11-16] shell | project switcher and boot hydration

The studio can now switch or create projects from the top bar instead of being
stuck on whatever the persisted store held. On boot, if the persisted project no
longer exists on this machine, the most recent server project is loaded (never
created implicitly). The Playwright smoke opens the project menu and sees the
server list. Verified: 23 vitest, 1 Playwright, tsc clean, eslint 0 errors.

## [2026-11-16] e2e | real-backend CRUD round trip

Added the missing Add shot action and a second Playwright test that runs a real
backend on an isolated temp DB/projects dir: create project via API, open
`/studio`, add a shot, edit the prompt, save, clear localStorage, and reload with
the shot still present. Each run uses a unique DB path so Playwright's startup
order cannot delete a live database. CI now installs the frontend/browser in the
backend job and runs both E2E tests; `MILIMO_E2E_PYTHON` overrides the interpreter.
Verified: 2 Playwright, 41 pytest, 23 vitest, tsc clean, eslint 0 errors.

## [2026-11-16] e2e | export leg and deterministic run dirs

Added the export E2E: create a project via API, write a real 1s clip into its
generated dir, save a shot, open Deliver, queue an export, and wait for the
completed output link. E2E paths are fixed and cleaned by a pre-Playwright
`node e2e/clean.mjs`, so a startup-order race can no longer leave SQLite without
tables; the local venv path is detected from the repo root. Boot hydration now
picks the most recently updated project. Verified: 3 Playwright, 41 pytest,
23 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f0 | resizable inspector, skip link, keyboard clips, axe

Inspector width is now pointer-resizable and persisted. Added a skip link and
focusable main region; timeline clips and trim handles are keyboard reachable
(Enter/Space select, Delete ripple-delete, arrow-key trim). Added Playwright axe
coverage (WCAG A/AA, zero critical) plus a keyboard surface-navigation test, and
deduplicated API stubs across E2E specs. Verified: 5 Playwright, 23 vitest, tsc
clean, eslint 0 errors.

## [2026-11-17] f0 | generated API types, Zod contracts, attribution

Checked in `web-app/openapi.json` + generated `src/api/schema.d.ts` with a
`npm run gen:api` script and a CI drift check. Added Zod runtime contracts for
health, plan, review, and share payloads; studio hooks now reject contract drift
instead of silently shaping unknown JSON. Added the Kubona Apache-2.0 attribution
file. Verified: 27 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f1 | multi-select, marquee, resize, align/distribute

Board authoring gained a multi-selection model (primary + set), Shift-click
toggles, Shift-drag marquee selects intersecting cards, dragging any selected
card moves the whole set, cards have a resize handle with live width/height, and
the toolbar can align left/center/right/top/middle/bottom and distribute H/V for
3+ cards. Verified: 28 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f1 | collapsible areas that carry their cards

Areas persist a `collapsed` flag (backend schema + board payload). Collapsing
hides contained cards and their links; expanding restores them. Dragging an area
moves every card whose center sits inside it, with positions committed through
the same revisioned layout save. Verified: 41 pytest, 28 vitest, tsc clean.

## [2026-11-17] f1 | editable shot list, bulk actions, links

The list equivalent now supports multi-select checkboxes, select-all, bulk
delete with confirmation, inline prompt and shot-type editing, and a typed-link
list with source → target and remove actions. Board and list selection share the
same multi-selection state. Verified: 28 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f1 | board undo/redo and pin control

Added a board-scoped history (separate from document undo) with undo/redo
buttons. Moves, resizes, area moves/resizes/creates, links, align/distribute,
arrange, pin toggles, and reset push one snapshot per gesture/action. Added the
missing pin/unpin control for a single selection. Board undo restores positions,
pins, areas, and links without touching document edits. Verified: 29 vitest,
tsc clean, eslint 0 errors.

## [2026-11-17] f1 | persisted, editable script view

Project-level script text now round-trips through ProjectState (`script_content`)
and the save payload. The studio Script view is an editable monospace editor with
word/shot counts, save-on-blur + explicit save, and a confirmed
parse-and-replace action that uses the existing local parser and storyboard
commit. Parsing warns that unmatched shots are removed. Verified: 42 pytest,
29 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f2 | take compare and stage guides

Edit surface now compares the selected take against the previous take with
side-by-side and a draggable wipe, both synced to the same clip-local frame and
playback state. Added toggleable grid, safe-area, and rule-of-thirds guides over
the stage. Compare buttons disable until a second take exists. Verified: 29
vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f2 | clip rate and slide

Added per-shot playback speed (0.25–4×): persisted, mapped through the document
store, reflected in timeline duration, and rendered with `setpts/speed` plus
`atempo` for embedded/standalone audio. Timeline gained Slide ◀/▶ for reordering
adjacent V1 clips. Added a real FFmpeg test proving 50 frames at 2× renders as a
1s file, plus filtergraph and round-trip tests. Verified: 44 pytest, 29 vitest,
tsc clean.

## [2026-11-17] f2 | waveforms, audio fades and solo

A1 clips render decoded waveforms on a canvas with a module-level peak cache.
Per-clip fade-in/out (seconds) persist and reach the audio filtergraph as
`afade`; the render honors track solo (soloing one track mutes the rest) and the
timeline labels expose M/S toggles. Added filtergraph tests for fades and solo.
Verified: 46 pytest, 29 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f2 | caption import and burn toggle

Added SRT/WebVTT parsing into sanitized frame cues, a timeline Import action, and
a project-level `captions_burn` flag (persisted, default true). Deliver exposes a
Burn captions toggle; the render skips ASS generation when it is off. Parser
tests cover SRT, VTT, HTML tags, and invalid input. Verified: 47 pytest,
32 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f2 | scale, pan and volume keyframes

Extended motion keyframes beyond opacity: scale, x/y pan, and volume are
sanitized, persisted, and interpolated. The render uses `zoompan` for
scale/pan, `geq` for alpha when opacity varies, and `volume=eval=frame` for
audio curves. Inspector rows expose each property at the playhead. Added
filtergraph tests plus a real render exercising opacity+scale+pan. Verified:
49 pytest, 32 vitest, tsc clean.

## [2026-11-17] f2 | render range and preset persistence

RenderSettings accepts start/end frames; the render clamps the range, seeks the
output with `-ss`, and reports progress against the range duration. Deliver has
in/out seconds with a full-cut reset, and the selected preset persists on the
project (`render_preset`). Added a real FFmpeg range test (2s clip, 1s export).
Verified: 50+ pytest, 32 vitest, tsc clean.

## [2026-11-17] f2 | timeline table and keyboard edit commands

Timeline gains a Table view listing every clip with editable trim in/out, speed,
mute, and delete, plus a track/start/duration readout. Global edit-surface keys:
B blade at playhead, R ripple delete, [ / ] slide the selected V1 clip. Keys are
ignored while typing or with modifiers. Verified: 32 vitest, tsc clean, eslint 0
errors, build passes.

## [2026-11-17] f2 | isolate the playback clock

Removed currentTime from the studio selection hook and from EditSurface /
StudioTimeline outer subscriptions. Media sync and active-clip changes now run
through a store subscription; only a tiny playhead and timecode subscribe per
frame. Inspector motion-frame readouts moved into a MotionPanel child. This
removes the previous whole-surface re-render storm at playback. Verified:
32 vitest, tsc clean, eslint 0 errors, build passes.

## [2026-11-17] f3 | ingredient composer, preview, batch queue, favorites

Generate surface now toggles real master references into a shot, inserts @trigger
mentions, exposes advanced params (negative/seed/frames/resolution/CFG/pipeline/
enhance/upscale), queues all pending shots through the batch endpoint, and shows
live active jobs with cancel. Preview-first generates a still, polls the job, and
attaches it as frame-0 conditioning before video spend. Takes can be favorited
(`favorite_take_id`) and compare prefers the favorite. Verified: 51 pytest,
32 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f3 | inpaint flow and batch matrix

Edit surface captures the current video/image frame, uploads it locally, and runs
the existing SAM-text + Flux inpaint endpoint as a shot-scoped job; the result is
selected as a new immutable take. Generate gained a batch matrix with per-shot
status/progress and generate/cancel actions. Inpaint jobs now carry `shot_id`
and appear in both shot and project take history. Fixed a test id collision.
Verified: 49 pytest, 32 vitest, tsc clean, build passes.

## [2026-11-17] f4 | saveable plan recipes

Command dock can save a plan as a named recipe with its mode and text, load it
against the current selection (supporting a `{{selection}}` placeholder), and
delete recipes. Recipes persist in the editor store. Added store tests. Verified:
33 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f4 | ranked local search

Library gained a Search tab that ranks shots, assets, master references, and
takes with deterministic token/prefix scoring — no embedding service, no network.
Results show score and jump to the shot where applicable. Added unit tests for
ranking and prefix behavior. Verified: 36 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f4 | LoRA list and activation in Setup

Setup now switches between base models and LoRAs, filtering by pipeline. LoRAs
use the real activation endpoint (which enforces an active base) and show real
status, size, and VRAM estimates. Verified: tsc clean, eslint 0 errors.

## [2026-11-17] e2e | reviewer @Agent to approved plan

Added a real-backend Playwright flow: create project/shot, mint a share link,
open the reviewer page, comment `@Agent prompt: …`, approve the shot, open the
editor, use "Plan fix in command dock", plan, and approve the patch. The shot
prompt updates through the undoable document path. Verified: 6 Playwright
passes.

## [2026-11-17] f4 | review take compare

Version stack entries can now be compared against the selected take in Review
with a synced wipe and stop control. This closes the review compare gap without
leaving the surface. Verified: 36 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] db | adopt an Alembic baseline

Added `backend/alembic.ini`, env, and a `0001_baseline` revision built from
SQLModel metadata. `init_db()` now runs Alembic: fresh DBs upgrade the baseline;
legacy DBs get the idempotent column repair, are stamped, then upgrade. Kept
create-all/repair as a safety net until future revisions exist. Added migration
tests for the stamped version and current columns; verified fresh and legacy DB
paths manually. Verified: 51 pytest.

## [2026-11-17] render | pixel-level golden assertions

Added real render pixel tests: a red→blue V1 cut samples the correct dominant
channel in each window, and a 50% V2 overlay produces a blended RGB frame. This
gives frame-level evidence beyond duration/streams/filtergraph. Verified:
53 pytest.

## [2026-11-17] jobs | one-shot OOM retry ladder

Queued video/image jobs now detect torch/MPS OOM messages, clear accelerator
caches, reduce resolution by 25% (8px aligned) and video frames to an 8k+1 count,
update the shot's real params, broadcast the retry, and rerun once. Attempts are
bounded by the durable ledger. Also fixed a double-decrement in the GPU wait
counter. Added unit tests for detection and retry math. Verified: 55 pytest.

## [2026-11-17] ops | request IDs, metrics panel, storyboard archive, async AI parse

Every HTTP request now gets an X-Request-ID and a structured timing log. Setup
shows the 7-day job error budget (status counts + recent failures) from
`/metrics/jobs`. Storyboard commit archives the pre-commit scenes/shots on the
project and exposes a download endpoint; the Script surface can download the
backup. AI script parsing now runs off the event loop via `asyncio.to_thread`.
Verified: 56 pytest, 36 vitest, tsc clean.

## [2026-11-17] perf | timeline viewport culling

Timeline clips outside the horizontal viewport (plus overscan) are no longer
mounted. Added a pure culling helper with a 200-clip unit test; the table view
still lists every clip. Verified: 38 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f0 | state skeleton: jobs, agent, document boundary, Query

Added TanStack Query around the studio shell and moved health polling onto
`useQuery` with contract validation. Introduced isolated job state (active jobs +
error budget), agent plan history, and a typed document-store adapter over the
legacy store so the eventual document migration is one file. Command plans now
record into the agent store. Verified: 39 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f3 | queue position and local estimate

`GET /projects/{id}/active_jobs` now merges the durable queued ledger with
in-memory running jobs, returning a queue position and a rough workload-based
seconds estimate. Generate shows both. Added a backend test with a queued video
job. Verified: 57 pytest, 39 vitest, tsc clean.

## [2026-11-17] f5 | shorts candidate workspace

Added a Shorts surface with a real local pipeline: import SRT/VTT (or fall back to
shot text/timing), build 10–45s candidate windows with a hook/density/numbers
score, select a candidate, choose 9:16 / 1:1 / 4:5, and export that range with
transcript captions burned in. Recent exports read the same render history.
Added candidate unit tests. Verified: 42 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f5 | pipeline lens

Generate gained a read-only pipeline lens: node summary (prompt, ingredients,
model, frames, resolution, upscale, take), validation warnings including the
LTX 8k+1 frame rule, a copyable reproducible recipe JSON, and a plan-API cost
estimate. No cloud calls. Verified: 42 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f5 | blocking lens

Edit gained a top-down blocking lens: draggable camera/character/object nodes,
camera rotation, labels, add/remove/reset, and shot context. Positions persist
on the shot as sanitized `blocking` JSON. Added backend round-trip/clamp tests.
Verified: 58 pytest, 42 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] contracts | typed ProjectState

Replaced `List[dict]` in ProjectState with typed `ShotState` and `SceneState`
models (extra fields allowed for forward compatibility). Save uses
`exclude_unset` so partial clients still patch rather than overwrite with
defaults. Regenerated OpenAPI/client types. Verified: 58 pytest, tsc clean.

## [2026-11-17] f5 | lazy 3D previz

Added a lazy-loaded three.js previz modal: floor/grid stage, blocking characters,
objects, camera bodies, camera path across shot order, orbit controls, and a PNG
snapshot. WebGL failure is caught and fails closed with an explanation. Three is
code-split into its own chunk so the editor does not pay for it at boot. Verified:
42 vitest, tsc clean, eslint 0 errors, build passes.

## [2026-11-17] f5 | Yjs collaboration spike

Added an opt-in Collab toggle for the storyboard: Yjs maps for card positions and
pins, BroadcastChannel transport between tabs on the same machine, remote
transactions applied without echo loops. No server/cloud. Added a two-doc
convergence test with a fake channel. Verified: 43 vitest, tsc clean.

## [2026-11-17] ci | coverage reporting

Added pytest-cov to requirements and CI runs `pytest --cov=backend` with a term
report. New modules land above 80% (planner 89%, review 89%, share 86%,
render_utils 95%, captions 90%, alembic baseline exercised); total repo coverage
remains 37% because the legacy GPU/model modules are not unit-testable without
weights. Ignored local coverage artifacts.

## [2026-11-17] status | remaining-plan closure pass

Updated the execution, frontend, and audit tables after the full implementation
pass: Alembic/OpenAPI types/OOM retry/request IDs/pixel tests/storyboard archive
are adopted; F0–F4 are mostly done, F5 is partial (shorts, pipeline lens, blocking
lens, lazy 3D previz, Yjs spike). Explicit remaining gaps are recorded instead of
being smoothed over: document-store migration, visual baselines, 60fps
measurements, mask/tracking/depth overlays, Tauri packaging, and parser queueing.

## [2026-11-17] f2 | tracking overlay and effects tab

Edit can now launch the existing object-tracking engine directly over the studio
stage with the live video ref and measured stage size, so masks overlay the cut
instead of handing off to the legacy editor. Inspector gained an Effects tab
summarizing speed, keyframes, fades, gain/mute, with edit shortcuts and a safe
reset. Verified: 43 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f1 | board note cards

Added draggable, editable, deletable note cards on the board, created at the
viewport center. Notes are included in board-scoped undo/redo and the persisted
editor layout. Added store tests. Verified: 44 vitest, tsc clean.

## [2026-11-17] f4 | active LoRA stack strength

Setup reads the active LoRA stack for the video slot and exposes a per-LoRA
strength slider that PATCHes the live stack (0–200%) after pointer-up. Activation
refreshes the stack. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] gen | queued AI script parsing

AI parsing no longer loads the text encoder inside the HTTP request. A new
`/storyboard/ai-parse-jobs` endpoint creates a Job and dispatches through the
durable GPU queue; the task loads the model, parses off-thread, writes a JSON
result into the project workspace, and broadcasts completion. The Script view can
queue it, poll, and commit the result. Added an endpoint test with enqueue
stubbed. Verified: 59 pytest, tsc clean.

## [2026-11-17] f5 | Tauri 2 desktop shell

Added a Tauri 2 scaffold for the same studio shell: config, Rust entrypoint,
capabilities, generated icons, `@tauri-apps/cli`, and a native `save_text_file`
command with the dialog plugin. Storyboard backups and pipeline recipes now use a
native Save dialog in Tauri and a download in the browser. `cargo check` and
`tauri build --debug --no-bundle` both succeed, producing a desktop binary.
The heavy sidecar/backend packaging remains a follow-up.

## [2026-11-17] perf | measured 500-card and 200-clip interaction

Added Playwright perf checks against a real 500-shot board and 200-clip timeline.
Only 11 cards and 27 clips mount; headless (frame-limit disabled) measured
57.8 fps pan and 274.9 fps timeline scroll. Also narrowed editor-store
subscriptions in TopBar/Inspector/CommandDock/StorySurface/StudioApp and removed
board geometry/camera from localStorage persistence (the server board/view
endpoints own those), eliminating redundant serialization during pans.

## [2026-11-17] a11y | zero serious axe violations

Raised muted/faint text contrast to pass WCAG AA, removed the nested interactive
role from the card resize handle, and upgraded the axe gate to fail on serious as
well as critical violations. Studio shell now reports zero violations under the
WCAG A/AA rule set. Excluded Tauri build output from ESLint. Verified: 44 vitest,
tsc clean, eslint 0 errors, axe 0 violations.

## [2026-11-17] test | visual regression baseline

Added a 1440×900 Playwright visual baseline for the studio shell with a
platform-neutral snapshot path and 20% pixel tolerance, plus deterministic
minimap hiding and card-visible waits. Verified locally against the committed
baseline. E2E count now includes shell, CRUD, export, reviewer-agent, axe,
keyboard, perf board/timeline, and visual.

## [2026-11-17] f1 | master-reference cards on the board

Added an opt-in Refs layer that renders draggable element/reference cards with
image, name, trigger, and type. Positions persist in the editor store; the layer
resets with board layout. This extends the board beyond shot cards without
changing sequence semantics. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] f1 | review task cards on the board

Open review comments and change-request approvals now appear as small task cards
beside their shot card. Clicking one selects the shot and opens Review. Tasks
respect collapsed areas and use the live review contract. Verified: 44 vitest,
tsc clean, eslint 0 errors.

## [2026-11-17] docs | operating manual, user guide, and final status

Rewrote AGENTS.md for the current architecture: `Milimo Video/` wiki layout,
legacy-vs-studio split, store boundaries, Alembic/typed contracts, durable queue,
render truth, studio capabilities, verification matrix, and explicit known gaps.
Replaced the user guide with a Studio-first manual covering launch, every surface,
keyboard map, generation/takes/inpaint, editing/tracking/blocking/3D, review/share,
agent plans, search, deliver, shorts, setup, privacy, troubleshooting, and limits.
Updated plan/frontend/audit tables to the measured state. AGENTS.md and the user
guide are now tracked.

## [2026-11-17] deliver | pad/crop reframe for variants

RenderSettings gained `fit: pad|crop`. Crop scales to cover and center-crops to
the output frame, so shorts/vertical variants are true reframes rather than
letterboxes. Deliver exposes Pad/Crop; Shorts always crops. Added filtergraph and
real FFmpeg tests asserting a 180×320 output from a 320×180 source. Verified:
60 pytest, 44 vitest, tsc clean.

## [2026-11-17] f5 | previz depth snapshot

3D previz can render a depth pass of the blocking scene (MeshDepthMaterial to an
offscreen render target, readPixels, vertical flip, PNG download). This gives a
real depth artifact from the current stage placement; the editor still has no
video-depth overlay source. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] state | separate studio and legacy document stores

The studio now owns its document/save/undo store (`milimo-studio-document`),
built from the shared slices but isolated from the legacy editor store. All studio
imports use `useStudioDocumentStore`; PlaybackEngine has a studio-specific clock.
Both routes talk to the same backend contract; switching routes no longer shares
in-memory document state. AGENTS/user guide/roadmap/audit updated. Verified:
tsc clean, 44 vitest, 9 Playwright (CRUD/export/reviewer path included).

## [2026-11-17] verify | final closure matrix

Final gates after the document-store split, crop reframe, tracking, effects,
reference/task cards, LoRA strength, queued AI parsing, Tauri shell, axe and perf
work: 61 pytest, 44 vitest, 9 Playwright, tsc clean, eslint 0 errors, Vite build
pass, verify_setup 0 failures, axe zero serious/critical, 500-card pan 57.8 fps,
200-clip scroll 277.9 fps. Working tree clean.

## [2026-11-17] library | direct studio upload

Library now uploads images, video, audio, and caption files through the existing
sanitized `/upload` endpoint with project scoping, then refreshes assets. Removed
the legacy-only empty-state handoff. Added a real-backend Playwright upload test.
Verified: 61 pytest (backend unchanged), 44 vitest, 10 Playwright.

## [2026-11-17] shorts | export all reframe variants

Shorts can export the selected candidate at 9:16, 1:1, and 4:5 in one action, each
as a true crop reframe. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] lens | editable pipeline app mode

Pipeline lens is no longer read-only: prompt, negative, frames, seed, resolution,
model, enhance, and upscale are editable, validation and estimate update from the
draft, and Apply writes the recipe back to the shot through the undoable document
store. Recipe JSON export/download remains. Verified: 44 vitest, tsc clean.

## [2026-11-17] f2 | live caption preview on the stage

Edit now overlays the active caption cue on the program monitor when caption
burn-in is enabled, using the same frame range as the render. The overlay follows
the playback clock with a tiny subscriber, so the rest of the surface stays
isolated. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] shell | delete project and save rename

Project menu can delete the current project with confirmation; project name
commits on blur. This closes the remaining basic project-management handoff to the
legacy editor. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] verify | post-split full matrix

After the studio/legacy document-store split, crop reframing, tracking, lens app
mode, caption preview, uploads, and project management: 61 pytest, 44 vitest,
10 Playwright (shell, CRUD, export, upload, reviewer-agent, axe, keyboard, perf
board/timeline, visual), tsc clean, eslint 0 errors, Vite build pass,
verify_setup 0 failures. Board pan 57.8 fps; timeline scroll 230.7 fps.

## [2026-11-17] ops | project integrity scan

Added `/projects/{id}/integrity`, reporting missing video/thumbnail files, orphan
shot→scene references, missing sequence indices, empty scenes, orphan board
nodes/links, and invalid caption ranges. Setup has a "Scan project" panel with
severity counts and the issue list. Added backend coverage. Verified: 44 vitest,
tsc clean.

## [2026-11-17] desktop | backend supervisor

Tauri exposes `start_backend`: it locates `run_backend.sh` from `MILIMO_REPO_ROOT`,
the current directory/ancestors, or the app resource dir, then spawns it with logs
written to `backend.desktop.log`. The studio's offline top-bar pill becomes a
"Start backend" action on desktop. `cargo check` passes; bundled Python sidecar
packaging remains open. Verified: 44 vitest, tsc clean, eslint 0 errors.

## [2026-11-17] a11y | all surfaces audited

Extended axe coverage across Generate, Edit, Review, Library, Deliver, Shorts, and
Setup. Fixed semantic meta grids (`dl`), unlabeled prompt fields, icon-only
timeline zoom buttons, and nested trim-handle interactivity. Every studio surface
now has zero serious/critical WCAG A/AA violations. Verified: 44 vitest, tsc
clean, 3 axe tests.

## [2026-11-17] shorts | export from an explicit source asset

Render accepts `source_path`, letting Shorts cut candidates directly from an
uploaded long video instead of only the project cut. The source is probed for
duration, rendered through the same filtergraph/range/crop path, and validated by
a real FFmpeg test (2s source, 1s vertical export). Shorts has a source selector
over uploaded video assets. Verified: 44 vitest, tsc clean.

## [2026-11-17] library | master-reference CRUD in studio

Studio Library now creates, edits, and deletes master references instead of
pointing at the legacy Elements view. The editor covers name, type, trigger word,
description, and an uploaded reference image. Backend trigger words are normalized
to one leading `@`. Added element CRUD contract tests and a Playwright create/edit/
delete path. Verified: 67 pytest, 44 vitest, 12 Playwright, tsc clean, build pass,
`verify_setup` 0 failures.

## [2026-11-17] polish | shorts ranges, recipe import, sidecar hook

Shorts candidates now expose editable in/out seconds with validation and reset
before export. The pipeline lens can import a saved recipe JSON (normalized
resolution/frames/model fields) back into its draft. The Tauri supervisor prefers a
bundled `milimo-backend` binary when present, then falls back to `run_backend.sh`;
a Python sidecar is still not built. Added recipe parser tests. Verified: 68
pytest, 47 vitest, 17 test files, tsc clean, build pass, cargo check.

## [2026-11-17] shell | surface rollback flags and current guidance

Implemented F0-05 rollback flags: `VITE_STUDIO_ENABLED=false` serves the legacy
editor at `/studio`, and `VITE_STUDIO_DISABLED_SURFACES` hides individual
surfaces while preserving their data. ModeRail, keyboard surface switching, the
active-surface guard, and the root route all honor the flags. Added `.env.example`
and parser tests. Removed stale "use the legacy editor" guidance now that Studio
owns script import, audio uploads, exports, tracking, masks, and inpaint.
Verified: 50 vitest, tsc clean, eslint 0 errors, build pass.

## [2026-11-17] deliver | persisted encoding controls

Completed P2-07: projects persist CRF, encoder speed, audio bitrate, and loudness
normalization. Deliver exposes all four, sends them to the render API, and keeps
them with the project. The filtergraph applies single-pass loudnorm after the
audio mix. Added Alembic revision 0002 (column-safe for fresh and legacy DBs),
round-trip/clamp tests, filtergraph tests, and a real ffprobe assertion for a
256k normalized export. Verified: 71 pytest, 50 vitest, tsc clean, build pass.

## [2026-11-17] desktop | native open dialog

Closed the missing native-open half of F5-05: Tauri now exposes `read_text_file`,
and `openTextFile` uses the dialog plugin with a browser file-picker fallback.
Pipeline lens recipe import uses the native picker on desktop. Added desktop
tests. Verified: 52 vitest, tsc clean, eslint 0 errors, cargo check.

## [2026-11-17] ci | enforce the JS bundle budget

Added `npm run check:bundle`: it fails when the built JS total exceeds 2 MB or a
single chunk exceeds 800 KB. Current build is 1,518 KiB across 3 chunks, largest
564 KiB. Wired the check into CI after `npm run build` and the verification matrix.
Verified locally: build pass, bundle check pass.

## [2026-11-17] correctness | fix undefined names and gate ruff

Ruff's correctness rules found real defects: `/assets/{id}` deletion called
`shutil` without importing it, Flux image completion referenced `out_path` and
`out_filename` outside their closure (the Asset record silently failed), and the
chained runner had duplicate imports plus a `del`/closure pattern. Fixed all of
them, removed a redundant LoRA list comprehension, added asset-directory deletion
coverage, and added `ruff.toml` enforcing E9/F63/F7/F82/F811. CI now installs ruff
and runs it as a blocking step. Verified: ruff clean, asset delete test passes.

## [2026-11-17] correctness | bind loop closures in chained generation

Enabled flake8-bugbear B023 and bound the chunk index, encoded video/audio, output
path, and chunk count as defaults in the chained runner. The callbacks already ran
before the loop advanced, but late binding was fragile. Verified: 72 pytest, ruff
clean, compileall.

## [2026-11-17] correctness | clean dead locals and restore thumbnail fallback

Enabled F841. Removed dead chained/render locals, dropped an unused Flux generator
(seed is already applied globally), and stopped discarding model-loader results.
Fixed a real wiring gap: `/uploads` computed a fallback thumbnail URL but never
assigned it, so generated videos without `thumbnail_path` showed no image. Added
coverage. Verified: 73 pytest, ruff clean, compileall.

## [2026-11-17] correctness | retain fire-and-forget tasks

Enabled RUF006 and kept strong references for the startup resume task, GPU queue
dispatch, and downloader progress broadcasts. Fire-and-forget `create_task` calls
can be garbage-collected before completion; they now live in tracked sets and are
discarded on done. Added a queue task-lifetime test. Verified: 74 pytest,
compileall, ruff clean.

## [2026-11-17] async | move ffmpeg helpers off the event loop

Enabled ASYNC221. Stitching, duration probes, thumbnail generation, last-frame
extraction, and source probing now run through `asyncio.to_thread` instead of
blocking the loop inside async routes/tasks. Added a real last-frame extraction
test. Verified: 75 pytest, ruff clean, compileall.

## [2026-11-17] generate | start/end frame conditioning

Preview stills can now be attached as frame 0 or the final frame, replaced per
frame, removed individually, and are listed with their frame index. Preview
generation still attaches frame 0 automatically. Verified: 52 vitest, tsc clean,
eslint 0 errors.

## [2026-11-17] deliver | cancel in-flight renders

Render history now exposes Cancel for queued/rendering/processing jobs and
refreshes status after the cancel request. Completes the F2-10 progress/cancel
path in the studio. Verified: 52 vitest, tsc clean, build + bundle check pass.

## [2026-11-17] exports | shared job cancellation

Added `useStudioCancelJob` and wired it into Deliver and Shorts render history so
both surfaces can cancel queued/running exports and refresh. Verified: 52 vitest,
tsc clean, eslint 0 errors.

## [2026-11-17] generate | persistent seed lock

Added `Shot.seed_locked` (default on) with Alembic 0003 and round-trip coverage.
The flag persists through save/reload, appears in Generate advanced parameters
and the Inspector, and gates the New take action so deterministic re-runs are the
default. Verified: 76 pytest, 52 vitest, ruff clean, tsc clean, build + bundle
check pass.

## [2026-11-17] queue | pause and resume dispatch

Added `GET /jobs/queue`, `POST /jobs/queue/pause`, and `/resume`. Paused dispatch
leaves queued jobs untouched, lets the current job finish, and resumes within
250ms. Health reports `queue_paused`; Setup exposes Pause/Resume. Added endpoint,
health-contract, and dispatch-gate tests. Verified: 78 pytest, 52 vitest, ruff
clean, tsc clean, build + bundle check pass.

## [2026-11-17] repair | project integrity repair

Added `POST /projects/{id}/repair`: clears missing media pointers, orphan scene
references, orphan board nodes/links, normalizes sequence order, and sanitizes
caption ranges without deleting files. Setup exposes Repair safe issues after a
scan. Added a round-trip repair test. Verified: 79 pytest, 52 vitest, ruff clean,
tsc clean, build + bundle check pass.

## [2026-11-17] setup | cancel model downloads

Setup now exposes Cancel download for active downloads (the API also removes
queued downloads). Added active, queued, and missing-download tests. Verified:
82 pytest, tsc clean, eslint 0 errors.

## [2026-11-17] verification | full matrix after remaining plan work

Re-ran the complete gate after the plan follow-ups: 82 pytest, 52 Vitest across
19 files, 12 Playwright (axe all surfaces, CRUD, export, references, review
agent, perf, visual), `compileall`, ruff correctness lint, tsc, ESLint 0 errors,
Vite build, 1,524 KiB bundle budget, `cargo check`, and `verify_setup` with 0
failures / 0 warnings. Measured 500-card pan ~58fps and 200-clip scroll ~279fps.

## [2026-11-17] docs | clarify remaining plan scope

Replaced the stale "deeper shorts polish open" note and expanded "Still open"
into explicit remaining/deferred items: bundled Python sidecar, legacy cutover,
the Phase 6 v2 lane, scoped-v1 gaps (transitions/LUT/titles/A2, embedding search,
template packs), and local visual baselines. No code changed.

## [2026-11-17] ingest | docs, README, and wiki refresh

Updated the public README, `docs/user_guide.md`, and the LLM wiki to the current
Studio + legacy architecture. The README now reflects React 19.2, the Studio
surfaces, the eleven-router/113-route backend, Alembic 0001–0003, durable queue,
render encoding, seed lock, desktop open/save, verification commands, and scoped
limitations. Wiki updates: overview, index, architecture, data-models, nle-timeline,
frontend-state, sse-eventing, studio-shell, engineering-audit remediation/product
gaps, getting-started, script-to-video, masking-tracking, and chained-generation.
Corrected stale claims about React 18, seven routers, 505-frame chaining, the
save endpoint, and a float32 transformer. No product code changed.

## [2026-11-17] cleanup | workspace and AGENTS refresh

Removed generated caches, test artifacts, and stale logs; left model weights,
databases, backups, projects, and user data intact. Updated AGENTS.md for React
19.2, the current routers/tasks/migrations, Studio state and rollback flags,
deliver/shorts/desktop capabilities, the explicit remaining/deferred gaps, and the
current verification matrix including ruff, bundle budget, OpenAPI drift, and
`cargo check`.

## [2026-11-17] audit | studio chrome, density, and responsive layout

Added [[systems/studio-ux-audit-2026-11]] with live Playwright measurements at
2048/1440/1280/1024/900/768. Confirmed: mode-rail labels all clip (53–85px labels
in 40px buttons), Inspector tabs are 486px in a 319px scrollbar-less scroller,
board toolbar is 1167px in 786/730px containers, and top-bar tabs overlap the
project switcher at 1024. Documented P0–P3 findings, missing UX functionality, and
F6 chrome-stabilization tasks; updated the frontend-refactor plan, implementation
plan, and wiki index.

## [2026-11-17] docs | link F6 audit from AGENTS

Added the F6 chrome-stabilization finding to AGENTS.md's remaining/deferred list
so agents do not treat the Studio layout as finished.

## [2026-11-17] feat | F6 studio chrome stabilization

Executed the F6 audit plan. Fixed the root cause first: the global `font: inherit`
shorthand on all controls was overriding component font sizes (rail labels were
rendering at 16px inside 40px buttons). Replaced it with `font-family: inherit`.

Implemented: responsive top-bar surface switcher under 1200px; grouped/wrapping
board and timeline toolbars with a More menu; 76px mode rail; Inspector primary
tabs + More; 11px+ type floor; 32px minimum controls; collapsible story spine,
Generate panes, Inspector, and command dock; command palette (`⌘K`), shortcut sheet
(`?`), agent dock shortcut (`⌘J`), comfort/compact density, card quick actions,
right-click shot menu, bulk selection bar, and auto-fit at narrow widths.

Verification: 0 clipped rail labels, 0 toolbar overflow, 0 Inspector tab overflow,
0 text nodes <10px, 0 buttons <28px at 2048/1440/1280/1024/900/768; 53 Vitest, 15
Playwright (responsive layout spec included), axe green, tsc/ESLint clean, build +
bundle budget pass. Visual baseline refreshed.

## [2026-11-17] fix | Studio media opening and preview

Deep-audited every media surface. Added a global `MediaPreview` (image/video/audio/
text) and wired Library assets/takes/references/search, Story cards (action,
double-click, context menu), board references, Generate takes/preview stills,
Inspector takes, Review rows/image, and Edit stage/pane. Fixed legacy absolute
localhost media URLs to route through the current API origin with tokens, classified
caption/text uploads correctly, and disabled pointer events on hidden card actions.
Added `media-preview.spec.ts` and caption-type coverage. Verified: 83 pytest, 56
Vitest, 16 Playwright, axe/tsc/ESLint/build/bundle clean.

## [2026-11-17] fix | video posters and hover playback

Added real poster frames for uploaded videos (`Asset.thumbnail_url`, Alembic
`0004_asset_thumbnail`), backfilled legacy video posters through `/uploads`, and
generated render posters. Added `MediaThumb` with poster-first cards, muted looping
hover playback, play badges, and fallback behavior, used across Library, Story,
Generate, Inspector, Review, Deliver, and Shorts. Verified: 84 pytest, 56 Vitest,
17 Playwright, axe/tsc/ESLint/build/bundle clean.

## [2026-11-17] audit | LLM configuration plane

Reviewed Kubona at `0dbf31c` (Apache-2.0): typed provider/wire/role config, local
discovery, provider verify/model catalog, revisioned redacted settings, sealed
keys, air-gap, budget guard, and the rich LLM Settings surface. Audited Milimo:
legacy-only Gemma/Ollama popover, runtime-only PATCH, no Studio panel, no
persistence/revision, no roles/fallback, no discovery/verify, no key posture,
hardcoded provider branches, and vision badges not wired to image input.

Filed [[systems/llm-settings-audit-2026-11]] with the provider/role design, API,
security rules, L0–L4 plan, and verification gates. Updated the implementation
plan, frontend roadmap, AGENTS, user guide, and index.

## [2026-11-17] implement | LLM configuration plane (L0–L3)

Executed the audit plan. Added the `app_setting` table with Alembic
`0005_app_settings`; built `backend/services/llm/` (`models`, `defaults`,
`prompts`, `secretbox`, `store`, `providers`, `runtime`, `discovery`) for typed
persisted settings with revision CAS, redacted reads, six providers, six roles
with ordered fallback, per-role sampling, loopback discovery, model catalog,
provider verify, async/sync transports, AES-GCM credential sealing, and air-gap
plus `MILIMO_LLM_CLOUD` enforcement. Replaced the in-memory `/settings/llm`
endpoints with `routes/settings.py`, rewrote `backend/llm.py` as a compatibility
facade, and removed every `config.LLM_PROVIDER` branch from video/chained/image/
storyboard tasks. AI script parsing now runs through `script_parse` and only loads
the LTX encoder when the role resolves to Gemma. Job params record
role/provider/model/latency provenance.

Studio Setup gained the AI & LLM panel: provider cards, list models, verify,
discovery + attach, role matrix with fallbacks, defaults, air-gap/cloud state,
save/revert/conflict, and a saved-config test prompt. The legacy popover now saves
through the revisioned `PUT` and links to Studio. Regenerated OpenAPI types.

Verification: compileall clean; 104 backend tests (17 new LLM tests); ruff clean;
verify_setup 0 failures; 61 Vitest tests (5 new panel tests); 20 Playwright tests
(3 new LLM specs incl. axe); tsc, ESLint 0 errors, Vite build, bundle budget; live
API check on a scratch DB including restart persistence and a 409 stale write.
L4 governance remains deferred. See [[systems/llm-settings-audit-2026-11]].

## [2026-11-17] audit | local-first LLM next steps

Tested the shipped plane against the machine's real local engines: Ollama
0.40.16 (`llama3.2:3b`, `granite4.1:3b`, `qwen3.6`, nomic embed), oMLX on 8787
(11 MLX models), and Strata on 47831 (Qwen3.8-Flash-Next, vision + speculative
decode). Discovery found all three; verify, role routing, live fallback, and
provenance worked end-to-end.

Measured: Ollama 3B enhancement 2.2 s warm / 7.4 s cold; oMLX 27B 23–25 s;
Strata completion 3.0 s warm; Strata script parse 31.7 s for 10 shots; oMLX 27B
cold load 36.2 s. oMLX ignores `stream: true` (0 bytes for 60 s); Strata streams
`reasoning_content` before content.

Filed [[systems/llm-local-first-next-2026-11]] with findings F1–F11 and phases
L5–L9: provider instances (oMLX + Strata collide on one `openai_compatible`
slot), capability truth (oMLX `max_model_len`, non-chat models, vision),
wrapper-safe prompt parsing (granite returned JSON to `prompt_image`),
discovery-seeded defaults (default `llama3.1` 404s here), warm-up/keep-alive/
timeouts, streaming negotiation, context preflight, local role eval, and
LLM/VRAM coordination. Cloud (L9) stays behind the local gates.

## [2026-11-17] implement | local-first LLM completion (L5–L9)

Implemented the whole local-first plan. Provider **instances** replace the single
type slot: `ollama`, `openai_compatible@127.0.0.1:8787`, and
`openai_compatible@127.0.0.1:47831` coexist, with a schema v2 migration and
attach-by-URL idempotence. Discovery now validates a model-catalog shape (so a
Docker console on :8080 is not offered), reports `needs_key`, and marks attached
instances. The catalog carries context (`/api/show`, `max_model_len`, `meta`),
parameter size, quantization, vision, and chat/embedding flags with separator-aware
matching.

Runtime: wrapper-safe enhancement parsing (granite's `{"prompt": …}` now yields
prose), discovery-seeded defaults with a mid-size instruct picker and explicit
unassigned roles, warm-up with measured load time, per-provider timeout/keep-alive,
streaming negotiation with a first-byte timeout and non-stream fallback,
context preflight, token/latency telemetry, and cancellation checks. Eval fixtures
for shot/image/script roles produce pass rate and p50, persist `last_eval`, and
drive an apply-recommended plan. Cloud now has a fail-closed daily budget with a
spend ledger, an eval gate for cloud role assignments, and sovereign/local/hybrid
profiles.

Generation: HTTP enhancement runs before the video pipeline loads and releases
every local role model afterwards; `llm_http` is a MemoryManager slot; image input
is sent over Ollama/OpenAI/Anthropic wires; takes show their enhancing model in the
Inspector; `verify_setup` reports engines, role readiness, and missing models.

Verified live against Ollama, oMLX, and Strata: seeded plan with real models,
capability catalogs, warm, streaming (Strata first delta 155 ms; oMLX streamed warm
and fell back when stalled), eval 100% (1.9 s / 6.4 s / 17.0 s), apply plan,
air-gap + eval-gate refusals, 131072-token preflight, unload to empty `/api/ps`,
and Strata vision describing a generated image. 125 backend, 61 Vitest, 21
Playwright, ruff/tsc/ESLint/build/bundle clean.

## [2026-11-17] docs | README, technical docs, and wiki sync

Brought the human-facing docs up to the implemented LLM plane. README gained a
"Local-first AI configuration" section, a twelve-router/133-route surface, the
`0005_app_settings` revision, the optional LLM environment variables, updated
verification counts (125/61/21), and the LLM caveats. The architecture doc now
diagrams the role runtime and local HTTP engines and adds a configuration-plane
component table; data models document the `app_setting` table, schema v2,
sealed keys, and spend ledger; AI pipelines replace the two-provider section with
the role runtime, residency/release rules, discovery/eval/budget, image wires,
and role-based script parsing; execution flow shows pre-pipeline enhancement and
the `script_parse` path; file dependencies map the settings router and
`services/llm/*`.

Wiki: data models add `appsetting`/`0005`, cloud-engines records the implemented
guardrails, the v2 LLM-slot recommendation points at per-role eval, and the
local-first page gains a file/route map. See
[[systems/llm-local-first-next-2026-11]].

## [2026-11-17] audit | generation notifications

Reproduced the reported failure: clicking Generate in Studio yields only a button
spinner. Root causes: Studio toasts are added to the document store but only the
legacy shell mounts a toast container (104 silent `addToast` call sites); Studio
never subscribes to `/events` (the global provider feeds only the legacy timeline
store); and `useStudioGenerate` has no progress/terminal reconciliation.

Live evidence: a Playwright probe showed `{"generating":1,"queuedToast":0,
"completed":0,"anyStatusRole":0}` and no batch toast after 1.2 s while SSE was
connected. An SSE listener saw a running job's denoise progress but no `queued`
or `started` events for a newly submitted job. Cancelling that pending job
returned `cancelled` with no event, `completed_at: null`, and the `QueuedJob`
row still counted in the queue.

Filed [[systems/job-notifications-audit-2026-11]] with the event-flow map, gap
register N-01…N-28 (toast layer, SSE bridge, queued/started/stage events,
terminal-once, persisted progress, pending cancel, payload consistency, replay,
accessibility, downloads/renders/parse/depth/inpaint), the proposed job event
envelope and catalog, and phases N0–N5 with gates. Updated
[[systems/sse-eventing]] to stop claiming Studio consumes events, the
implementation plan status, AGENTS known gaps, and the index.

## [2026-11-17] audit | memory management and model residency

Investigated the macOS out-of-memory failure. On this 128 GB machine (115.4 GB MPS
ceiling), Strata held **33 GB** and oMLX **6.5 GB** of model memory that Milimo
could not see, while a 121-frame 768×512 40-step upscale job began at 13:59. The
job denoised for 9:51, spent 1:52 + 0:53 in two 3-step upscale phases, then macOS
raised "your system has run out of application memory" at ~14:16. Swap reached
40–42 GB of 44 GB (92–96%). The worker died; the reloader parent was left spinning
at 91% CPU with no model, HTTP stopped responding, and the Job/QueuedJob stayed
`processing/running`.

Filed [[systems/memory-management-audit-2026-11]] with measured footprints (LTX
~76 GB, Flux ~36 GB, SAM 3, depth), the static 40/24 GB guard gap (real peaks
2–3× higher), missing external-engine accounting, activation/deactivation bypassing
the GPU lease, no idle TTL, no swap/pressure guard, no MPS cap, blocking encode, and
crash recovery gaps. Proposed engine adapters (Ollama `/api/ps`, oMLX
`/v1/models/status` + unload, Strata opaque), a byte budget with peak estimation and
leases, admission/downgrade, model-aware batching, observability, and phases
M0–M5. Updated [[concepts/memory-slots]], the implementation plan, architecture,
AGENTS engineering law/known gaps, the LLM residency notes, the notification event
catalog (`job.waiting`), and the index.

## [2026-11-17] implement | memory budget and job notification plane (M0/M1/M3/M4, N0–N4)

Shipped both audits end to end.

Memory: new `backend/services/memory/{models,costs,engines,budget,sampler}.py`.
`memory_manager` now admits by measured bytes, leases slots, refuses conflicts
that are in use, evicts idle slots on TTL/pressure, pins keep-warm, and exposes
status. Cost registry (LTX 76 GB base, Flux 36 GB, workspace/frames/resolution/
upscale) plus a physical-free, free-swap, and safety-margin gate — swap occupancy
alone no longer blocks work. Engine adapters: Ollama `/api/ps`, oMLX
`/v1/models/status` and unload, generic `memory_gb` override. Wired admission into
video/image/chained/ai_parse/inpaint/element, pipeline load, model activation,
and LLM warm/reserve. `/system/memory`, `/preflight`, `/unload`, `/keep-warm`;
`/settings/llm/memory` and provider unload; Setup `MemoryPanel`; Generate
preflight; per-job sampler persisted in `Job.memory_json`; MPS
`set_per_process_memory_fraction`; encode moved to a thread; uvicorn reload now
opt-in (`MILIMO_RELOAD=1`).

Notifications: `events.py` gained `seq` ids and `Last-Event-ID` replay; new
`job_events.py` emits the typed envelope and owns the single terminal write/emit.
`job_utils` emits queued/started/waiting/retrying/cancelling and persists
progress/stage; recovery fails interrupted jobs instead of relaunching. Pending
cancel clears `Job` + `QueuedJob`. Studio got a mounted `StudioToastLayer`, an
`ActivityCenter` with active/history/unread, StatusStrip stage/progress, the
`jobBridge` job→shot reconciler, typed SSEProvider with legacy mapping, inline
Generate progress, and resync/watchdog wiring for takes/assets/renders/models.

Alembic `0006_job_progress` adds `stage`, `started_at`, `memory_json`; LLM schema
v3 adds `memory_gb`. Tests: `tests/test_memory_budget.py` (25),
`tests/test_job_events.py` (9), `jobBridge.test.tsx` (6),
`JobProgressUi.test.tsx` (3). Green: 159 backend, 70 Vitest, 21 Playwright,
ruff/tsc/lint/build/bundle/verify_setup.

Live evidence: SSE showed `job.queued → started → loading → conditioning →
denoising`; a paused queued job showed `queued → waiting → cancelled` and
`/status` ended `cancelled`. `257f/1216×704/upscale` (166.7 GB) was refused with
the free-RAM/margin reason; a 121f denoise was cancelled at progress 11 with no
OOM and the 76 GB slot released on demand.

Updated both audits with a task-by-task status table, the roadmap open/deferred
list, AGENTS architecture/gaps/counts, and the index.

## [2026-11-17] implement | queue priority, load rollback, notification preferences

Closed the remaining P1 gaps that were still open in the two audits.

- `job_utils.job_priority` + `wait_for_turn`: dispatch order is decided before
  the GPU semaphore — interactive video/image (20), inpaint (30), element (50),
  ai_parse/depth (60), FIFO within a tier. Tests assert the head order and that a
  bulk job blocks behind an interactive one.
- `model_engine.load_pipeline` and `flux_wrapper.load_model` now clear the
  pipeline and release the slot when construction or the MPS VAE cast raises
  (M1-05).
- Studio `notifications/preferences.ts`: persisted toasts/OS/sound preferences,
  a Setup panel, browser Notification on hidden tabs, and a WebAudio cue.
  Terminal toasts are suppressed when disabled while history still records.
- Engine snapshot failures no longer break `/system/memory`; the active-job list
  is now replaced from `/active_jobs` so a missed terminal cannot leave a spinner.

Counts: 162 backend tests, 72 Vitest tests, 21 Playwright tests.

## [2026-11-17] implement | complete the remaining M2/M4/N4/N5 work

Finished the items still marked open after the first pass.

- **Waiting for memory (M2-03/M4-03)**: `AdmissionDecision` gained `retryable`
  and `downgrade`. A transient refusal raises `MemoryDeferred`, and
  `run_queued_job` emits `job.waiting`, clears `started_at`, stores
  `_memory_waits`/`_memory_not_before`, and re-dispatches with 10–120 s backoff
  (20 attempts) without holding the GPU semaphore. A permanent refusal applies
  the largest fitting preset before any load, persists it to Job/QueuedJob/Shot,
  and emits `job.retrying`.
- **Downgrade ladder (M2-04)**: drop upscale → frames 121/93/81/65/49 →
  768×512/640×448/512×320/384×256; images step 1024→512. Preflight returns the
  exact preset and estimate; Generate renders an Apply & generate banner.
- **Model-aware grouping (M2-02)**: `_queue_head` sorts by priority, then by the
  slot already resident, then FIFO, so Flux jobs group behind Flux and video jobs
  behind LTX.
- **Metrics (M3-04/N4-04)**: memory loads/unloads/evictions/refusals, queue wait
  p50/p95/max, and SSE dropped/peak-pending/last-broadcast-age on `/metrics/jobs`
  and `/system/memory`.
- **MPS peak (M4-02)**: decoded video/audio references and the executor tuple are
  released before encode with a cache flush; MPS already tiles decode.
- **N5**: per-stage ETA from the observed step rate, history Retry in the activity
  center, optional `MILIMO_JOB_WEBHOOK`, and a Tauri `notify` command using macOS
  Notification Center (`cargo check` clean).

Tests: 172 backend, 79 Vitest. Live preflight for 121f/768×512/upscale now
returns `ok:false, retryable:false` with `downgrade: upscale off (95.55 GB)`;
`/metrics/jobs` returns the new counters and empty histograms on an idle queue.

## [2026-11-17] fix | Studio pinned itself to the placeholder project

Found while validating the downgrade banner live: a server project with id
`default` (the local sentinel, auto-saved with "Untitled Project") was always the
newest row, so `useStudioProjectBootstrap` loaded it and ignored every real
project. Bootstrap now excludes `default` whenever any real project exists, keeps
an explicitly opened UUID, and de-duplicates the project list read. Added a
stable `generate-shot` test id to the main Generate button and tests for the
sentinel and the kept-project path.

The Playwright config no longer reuses an existing Vite server: a dev server on
:5173 pointed at API :8000 made the suite silently test the wrong database while
the e2e backend ran on :8010. It now fails loudly instead.

## [2026-11-17] implement | mute, window-title progress, and desktop notifications

Closed the last N5 polish items: per-project notification mute (persisted with
toasts/sound/OS), a Tauri `set_window_title` command so the desktop window shows
`% · stage · Milimo` while busy, and the macOS Notification Center `notify`
command. Also changed the e2e Vite server to `reuseExistingServer: false` so a
dev server pointed at :8000 can no longer make the suite test the wrong database.
Counts: 172 backend, 79 Vitest, 21 Playwright.

## [2026-11-17] implement | remaining M5 and dock badge

- **SAM 3 lazy load**: `start_sam_server.py` starts without loading; the first
  image request loads under a lock, `/load` warms, `/unload` releases with a
  cache flush, `/health` reports lazy/loading/error, and the memory manager
  releases SAM through `/unload`. Live check on :8011: startup health returned
  `model_loaded:false, lazy:true` and `/unload` returned clean without a load.
- **M5 comparison tool**: `backend/benchmark_models.py` emits a per-lane table
  (video 86.8 GB, image 43.0 GB, SAM 8.4 GB, depth 0.6 GB on the reference
  workloads) and can run a small job to record measured peak/load/wall into a
  gitignored `benchmarks/model-lanes.json`. Quantized/distilled/text-encoder
  swaps are blocked on model artifacts and documented as such.
- **Dock badge**: Tauri `set_badge` uses `set_badge_label` (macOS) and
  `set_badge_count`; the activity center shows the progress percent and active
  count, cleared on idle. `cargo check` clean.
- Tests: 181 backend, 79 Vitest, 21 Playwright.

## [2026-11-17] docs | sync wiki, technical docs, and README with M0–M5 / N0–N5

Updated the human-facing and technical sources after the memory and notification
implementation.

- README: added "Memory, scheduling, and progress"; architecture rows for the
  budget, events, and desktop progress; 13 routers / 139 operations; Alembic 0006;
  memory/event API and `/metrics` fields; p50/p95 queue metrics; SAM lazy load;
  opt-in reload; memory/webhook env vars; 175/79/21 verification; M5/desktop caveats.
- Wiki: rewrote [[systems/sse-eventing]] for the typed envelope/replay/terminal
  emitter and Studio surfaces; updated [[systems/architecture]] module map,
  topology, and gotchas; replaced the stale [[concepts/memory-slots]] audit notes
  with the implemented budget/lease/TTL/admission behavior; updated
  [[systems/data-models]] (`job.stage`/`started_at`/`memory_json`, v3 settings,
  `0006_job_progress`); updated [[systems/frontend-state]] (job store/bridge/
  preferences/bootstrap); updated [[models/sam-3]] lazy lifecycle and the roadmap
  phase tables (M0–M5, N0–N5) and [[overview]] counts.
- Technical docs: `01` (13 routers, typed SSE, lazy SAM, memory section), `02`
  (Job fields, QueuedJob, v3 settings, recovery), `03` (measured budget/admission/
  downgrade/scheduling diagram), `04` (typed SSE + Studio job state), `05`
  (ledger/typed events/startup/cancel), `06` (job_events/services/memory/routes),
  `CODEBASE_ANALYSIS`, `sam3-bible`, and the user guide (preflight safe preset,
  waiting for memory, memory/notifications panels, updated limitations).

## [2026-11-17] investigate | triage the five open GitHub issues

The issues are from March, before the memory/event implementation, so each was
checked against `main` rather than assumed current.

- **#4 AE 404 — fixed.** `flux2.util` now downloads the FLUX.2 AE from
  `black-forest-labs/FLUX.2-dev` (`ae_repo_id`); the manifest's nonexistent
  `FLUX.2-klein` repo and dead 9B AE entry were corrected, the AE is marked gated,
  and `load_ae` raises a clear error instead of `sys.exit(1)`.
- **#6 16 GB CUDA OOM — mitigated.** The budget now sizes against free VRAM
  (`torch.cuda.mem_get_info`), caps the process at `MILIMO_CUDA_MEMORY_FRACTION`,
  and uses the CUDA FP8 bases (40 GB video / 18 GB image). A 16 GB card is refused
  before load with a “VRAM” reason and the offload note; an 80 GB card admits the
  same job. CPU offload is still not implemented (no hooks in the vendored LTX
  pipelines).
- **#3 API — resolved by documentation.** Added `docs/07_api.md` with REST/SSE
  batch examples, preflight, auth, cancellation, and guardrails.
- **#2 quantized models — open.** No GGUF loader/artifacts; the concrete plan and
  the M5 benchmark baseline are recorded.
- **#1 Wan 2.2 — open.** A separate model family (DiT/VAE/umT5/LoRAs); the LTX
  adapters and chunk/audio math cannot be reused.

New page: [[systems/github-issues-triage-2026-11]]. Tests: 36 memory (incl. CUDA/
AE), full suite re-run.

## [2026-11-17] investigate | open-work inventory and resolution plan

Audited GitHub issues/PRs, CI, the roadmap deferrals, AGENTS gaps, and code
markers against `main`.

Findings:
- **CI is red on main/develop.** Backend tests fail with SQLAlchemy 2.1's
  timezone-aware datetime enforcement; reproduced in a clean venv with
  SQLModel 0.0.50 + SQLAlchemy 2.1.9. Local SQLModel 0.0.31 / SQLAlchemy 2.0.46
  passes 180 tests. Root cause is the unpinned `sqlmodel` in requirements.
- **PR #7 is superseded** (both `enhance_prompt` call sites already unpack).
- **PR #8 is still valid** (`encode_video` lacks `torch.no_grad()`; CUDA
  inference-tensor crash).
- **Issues:** #4 fixed, #3 documented, #6 mitigated (no CPU offload), #1/#2 are
  real feature gaps.
- **Deferred:** bundled sidecar, legacy cutover, NLE/search/pack gaps, M5
  artifacts, Wan, LLM caveats, per-job priority, tray icon, perf flake.

Filed [[roadmap/open-work-resolution-2026-11]] with P0 (green CI, PRs, issue
triage, two clean releases), P1 (CUDA/quantized, sidecar, cutover), P2 (product
gaps), P3 (quality/ops), dependencies, and definition of done. Indexed in the
wiki.

## [2026-11-17] execute | P0 of the open-work plan

- **CI root cause fixed.** Pinned SQLModel 0.0.31 + SQLAlchemy <2.1, the flux2
  vendor pins (safetensors/transformers/einops/accelerate), and FastAPI/Pydantic
  for stable OpenAPI. Added `pip check`. CI backend and frontend are green on
  `main` and `develop` at 6a36cc8.
- **PR #8 applied** (merge 28729c5): `encode_video` consumes the lazy generator
  under `torch.no_grad()` and detaches before CPU. Added
  `tests/test_encode_video.py`, which fails if frames are consumed with autograd.
- **PR #7 closed** as superseded.
- **Issues triaged**: #4 and #3 closed with evidence; #6/#2/#1 replied to and
  labeled, #6 kept open for CPU offload/16 GB support.
- **Dependabot/security**: grouped weekly config committed; vulnerability alerts
  enabled.
- **P1-1 decision**: CPU offload is not a config flag in the vendored LTX
  pipelines; the 19B FP8 transformer still exceeds 16 GB. Quantized checkpoints
  or a larger GPU are the path.
- **P1-2 status**: diffusers 0.36 has GGUF quantization but no Klein/Qwen3
  pipeline; the 5.7 GB Q4_K_M artifact is available but needs a 0.37.dev pipeline
  or an in-house dequant loader.
- Tests: 181 backend, 79 Vitest, 21 Playwright.

## [2026-11-17] research | model and workflow landscape, plan re-sequence

Searched current releases and production workflows and filed
[[research/model-landscape-2026-11]].

- **LTX-2.5** (Aug 11, 2026; 22B, native audio) has an official diffusers
  pipeline with `enable_model_cpu_offload()` and `vae.enable_tiling()`; support is
  not in a diffusers release yet (install from main). This is the real #6 path, not
  a vendored-pipeline patch.
- **Wan 2.2** is Apache 2.0 with a 5B TI2V model (~24 GB official, ~8 GB ComfyUI
  offload) and A14B/S2V/Animate variants; Wan 2.5+ is closed.
- **License constraints**: HunyuanVideo 1.5 excludes EU/UK/KR; MiniMax H3 excludes
  US/EU/UK/KR; Qwen-Image 2.1 is research-only. They cannot be bundled defaults.
- **Commodity lanes**: Qwen3-TTS and SeedVR2 are Apache 2.0; OTIO/FCPXML/Resolve
  XML and ComfyUI are the workflow interfaces the community uses.
- Re-sequenced [[roadmap/open-work-resolution-2026-11]]: P1 is now LTX-2.5
  diffusers/offload, Wan 2.2 5B, quantized Flux, CUDA validation, then packaging;
  P2 starts with OTIO/ComfyUI interop and the audio/upscale/Wan-family lanes.
  Updated [[roadmap/implementation-plan-2026-11]], the index, and README.

## [2026-11-18] implement | OTIO export, Wan 2.2, and quantized Flux

Executed the code-completable P1/P2 items.

- **OTIO export**: `utils/otio_export.py`, `GET /projects/{id}/otio`, Deliver
  "Export OTIO" button. V1 magnetic, free V2/A1 with gaps, trims, speed as
  LinearTimeWarp, MissingReference for offline media. Tests cover tracks, ranges,
  gaps, and missing media. Live endpoint returned a valid OTIO timeline.
- **Wan 2.2 TI2V-5B**: `models/wan_wrapper.py`, `tasks/wan.py`, manifest/root
  snapshot download support, `pipeline_override=wan` branch/UI, Wan costs. Real
  MPS generation: 17 frames 256×256 in 11.4 s after a 22.7 s load; a queued API
  shot completed with an H.264 MP4 + thumbnail (40 steps).
- **Quantized Flux**: diffusers 0.37.1 + gguf; `models/flux_gguf.py`,
  `MILIMO_FLUX_GGUF`, manifest Q4_K_M entry, GGUF guards for IP-Adapter/inpaint.
  Real MPS generation 256×256 verified (non-black, 6k colors); app-level path
  verified.
- Tests: 190 backend (OTIO 2, GGUF 2, Wan 5), 79 Vitest, 21 Playwright.
- Caveats: CUDA validation for Wan/GGUF, LTX-2.5 diffusers, image-to-video for
  Wan, XML/OTIO import, and the sidecar/cutover remain.

## [2026-11-18] implement | OTIO import, shot priority/steps, and desktop tray

- **OTIO import**: `utils/otio_import.py` + `POST /projects/{id}/otio/import`,
  Deliver **Import OTIO**. Rebuilds V1/V2/A1, trims, speed, and project media URLs;
  round-trip test. Alembic `0007_shot_priority_steps` adds `shot.priority` and
  `shot.num_inference_steps`.
- **Per-shot priority/steps**: persisted in the shot document, shown in Advanced,
  copied into queue params; `row_priority` lets an explicit priority override the
  kind default. Tests for persistence and queue order.
- **Tauri tray**: tray icon with show/quit, tooltip command wired to the activity
  center; `cargo check` clean.
- Tests: 193 backend, 79 Vitest, 21 Playwright.

## [2026-11-18] implement | FCPXML export

Added `otio-fcpx-xml-adapter` and `?format=fcpxml` to the OTIO endpoint, a Deliver
**Export FCPXML** button, and a round-trip test that reads the XML back through
OTIO. Tests: 194 backend, 79 Vitest, 21 Playwright.

## [2026-11-18] implement | ComfyUI bridge

- **Client**: `services/comfy.py` — API-format parsing/validation, per-node
  overrides, filename prefixes, `/prompt` submit, `/history` polling,
  `/view` streaming downloads, `/queue`/`/interrupt` cancellation, `/free`
  model release, and websocket progress (`execution_start`, `executing`,
  `progress`, `progress_state`, `executed`, terminal) with polling fallback.
- **Queue/task**: `tasks/comfy.py` + `kind="comfy"`; declared external residency
  (`MILIMO_COMFY_GB`), `comfy` memory slot, conflict/unload integration; every
  output streamed into `generated/`, inserted as an `Asset`, and the primary
  image/video attached to the shot.
- **API**: `/comfy/{health,validate,run,free}`; generated OpenAPI and TS types.
- **UI**: Setup ComfyUI bridge probe/free/open; Generate ComfyUI workflow panel
  with import, model-free smoke graph, validate, run, cancel, and localStorage
  draft.
- **Dependencies**: `websockets>=12`.
- **Tests**: parser/overrides/validation, fake ComfyUI server client, progress
  mapping, route errors, plus a live end-to-end `EmptyImage → SaveImage` run that
  asserts the copied file, asset row, shot update, and 100% terminal. Live probe
  on ComfyUI 0.45.0 (port 8001) completed and produced a PNG.
- **Docs**: `systems/comfyui-bridge.md`, index, architecture, memory slots,
  roadmap, README, user guide, AGENTS.
- Tests: 206 backend, 91 Vitest, 21 Playwright.

## [2026-11-18] implement | FCPXML import

The OTIO import route auto-detects XML and reads it with the `fcpx_xml` adapter,
so a Final Cut/Premiere round-trip returns shots too. Deliver labels the action
**Import OTIO/FCPXML**; a two-clip round-trip test asserts fps, frame counts,
trims, and project media URLs. Resolve XML remains (no adapter).

## [2026-11-18] implement | ComfyUI optionality and project LUT grading

- **ComfyUI is explicitly optional.** `MILIMO_COMFY_ENABLED=0` short-circuits
  `/comfy/health` (`enabled:false`, no network probe), rejects `/comfy/run` and
  `/comfy/free` with 409, hides the Generate workflow panel, and shows a disabled
  card in Setup. The Generate panel also stays hidden while ComfyUI is offline.
  Core boot, projects, memory preflight, generation, render, and interchange never
  touch `services.comfy`. `verify_setup` reports an offline/disabled bridge as a
  note, not a warning.
- **LUT grading.** Project `render_lut` (Alembic 0008), `.cube`/`.3dl`/`.dat`/`.m3d`
  uploads under `assets/lut/`, Deliver add/replace/remove, and
  `lut3d=...:interp=trilinear` on the V1/V2 composite before captions.
- **Verification.** Live servers with the bridge disabled and with an unreachable
  URL both served `/health`, created/loaded projects, and passed memory preflight.
  A real FFmpeg render through a red/blue swap LUT asserted the blue channel; a
  missing LUT returns 404.
- Tests: 206 backend, 91 Vitest, 21 Playwright.

## [2026-11-18] implement | titles, A2, and V1 crossfades

- **Titles**: `project.titles` (Alembic 0009) with Deliver add/remove, drawtext
  after LUT and before captions, escaped text/color/size/position. Real render
  test asserts burned-in pixels.
- **A2**: `track_index == 3` is a real second audio lane with its own mute/solo,
  Web Audio playback predicate, `amix` with A1/embedded audio, OTIO A2 track
  export/import, and timeline picker.
- **V1 crossfade**: `shot.transition_in` (Alembic 0010), shared overlap clamp in
  the timeline layout, render, and OTIO metadata; incoming alpha ramps 0→1 over
  the previous clip. Real pixel test asserts a red/blue midpoint mix.
- Tests: 217 backend, 91 Vitest, 21 Playwright.

## [2026-11-18] implement | lens recipe packs

The pipeline lens accepts a `{name, recipes}` object or a bare array, validates
every recipe, and lists them for selection; single recipes still apply directly.
Shorts template packs remain.

## [2026-11-18] fix | skip chunk-0 re-enhancement after pre-load pass

The chained task now consults `should_enhance_chunk`: when `_preenhance_http`
already rewrote the prompt before LTX loaded, chunk 0 is not enhanced again after
the pipeline occupies memory. Gemma still uses the in-pipeline path. Unit test
covers pre-enhanced, unenhanced, disabled, and later chunks.

## [2026-11-18] fix | abort non-stream LLM HTTP requests

`services.llm.providers.acomplete` now runs the request as a cancellable task and
polls `cancel_check` every 100ms; cancellation closes the socket instead of
waiting for the provider timeout. `runtime.chat` and the non-stream stream
fallback map `ProviderCancelled` to `LLMCancelled`. A fake slow-server test
asserts a sub-2s abort.

## [2026-11-18] docs | README, docs, and wiki refresh

Rewrote the README as a concise current overview (surfaces, engines, architecture,
quick start, optional ComfyUI, verification, limitations). Updated the system
architecture, data-model, execution-flow, API, user-guide, overview, and timeline
pages for Wan 2.2, optional ComfyUI, LUT, titles, A2, crossfades, OTIO/FCPXML,
lens packs, and the current 214/90/21 test baseline.

## [2026-11-18] implement | SAM 8002, LTX-2.5, thumbnails, asset delete, model recommendations

- **SAM** now defaults to **8002** (`MILIMO_SAM_PORT`), freeing 8001 for ComfyUI.
  Updated clients, memory sampler/unload, `run_sam.sh`, and docs. Live `/sam/health`
  reports online on 8002.
- **LTX-2.5** adapter (`models/ltx25_wrapper.py`, `tasks/ltx25.py`) uses diffusers
  `LTX2Pipeline`, CPU offload/VAE tiling, native audio, `pipeline_override=ltx25`,
  manifest/memory/UI wiring. Gated weights are not bundled; tests use a fake pipe.
- **Edit thumbnails**: V1/V2 timeline clips render the shot poster behind the title;
  stills fall back to the media URL.
- **Asset deletion**: Library cards expose Delete; the endpoint removes the file and
  poster, clears shot pointers, and marks jobs deleted. Live upload/delete verified.
- **Model recommendations**: manifest flags LTX-2.5, LTX-2, Wan 2.2, Flux 2 Q4,
  SAM 3, and Depth; Setup/Model Library show Recommended and rationale.
- **Docs**: detailed ComfyUI bridge guide, README screenshots, SAM port, LTX-2.5
  page, delete instructions, and current model lanes.
- Tests: 217 backend, 91 Vitest, 21 Playwright.

## [2026-11-18] implement | real LTX-2.5 download and MPS generation

Accepted the gated access request via the Hub ask-access endpoint and downloaded
the components referenced by `model_index.json` into
`backend/models/ltx-2.5-diffusers` (97 GB; duplicate full/merged shards skipped).
Upgraded `diffusers` to 0.40.0, `transformers` to 5.23.0, and `huggingface-hub`
to 1.33.0; refreshed the Flux package ranges.

The full pipeline loads in ~57s. The adapter needed an MPS workaround: diffusers'
LTX2 RoPE builds float64 frequencies when `double_precision` is set, so the MPS
path disables that flag. Real run: 9 frames at 256², 1 step, seed 42, `a slow
cinematic dolly-in...`; output was video with native 48 kHz audio. The queued
`pipeline_override=ltx25` API path completed, attached the shot, wrote a poster,
and `ffprobe` showed both video and audio streams. Tests: 218 backend.

## [2026-11-18] audit | ComfyUI frontend integration

Filed [[systems/comfyui-frontend-audit-2026-11]] and the phased
[[roadmap/comfyui-frontend-plan-2026-11]] (C0 trust/recovery → C1 binding/media
→ C2 library/templates → C3 results/batch → C4 remote/config → C5 polish).

Findings: the backend bridge is solid; the frontend is a paste-JSON runner. Live
evidence: `/comfy/validate` accepts an unknown `class_type`; ComfyUI exposes 2,082
node definitions and 11 workflow templates that the UI does not use. P1 gaps:
no shot/node bindings, no project-media upload, structural-only validation, global
localStorage draft, no output selection, no job recovery after reload. Updated the
P2 roadmap row, implementation plan, ComfyUI docs, and index.
