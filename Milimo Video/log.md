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
