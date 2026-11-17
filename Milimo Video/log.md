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
