# Milimo Video Wiki Index

Catalog of every wiki page. Updated on every change. The LLM reads this first to
navigate, then drills into pages.

- Last updated: 2026-11-18

## Overview
- [overview](overview.md) — what Milimo Video is: the AI-native cinematic studio,
  a professional NLE for AI filmmaking running entirely local-first.

## Systems
- [systems/architecture](systems/architecture.md) — the split architecture: React
  frontend, FastAPI backend, SAM 3 microservice, how they talk (REST, typed SSE),
  the memory/job-event plane, plus the module map (key files and import edges).
- [systems/nle-timeline](systems/nle-timeline.md) — the magnetic timeline: V1/V2/A1
  tracks, Zustand store, PlaybackEngine, snapEngine.
- [systems/data-models](systems/data-models.md) — SQLite/SQLModel entities
  (Project/Scene/Shot/Element/Asset/Job/AppSetting), Pydantic schemas, TypeScript
  types, Alembic revisions, and the LLM settings/spend documents.
- [systems/frontend-state](systems/frontend-state.md) — the Zustand 7-slice
  "God Store": slice inventory, persist/zundo config, selector discipline and
  perf strategies, server-sync via SSE.
- [systems/storyboard-engine](systems/storyboard-engine.md) — screenplay to
  shot list: regex vs Gemma AI parsing, ElementMatcher 8-signal scoring,
  concept-art thumbnails, generation dispatch.
- [systems/sse-eventing](systems/sse-eventing.md) — real-time eventing: typed
  `job.*` envelope, `seq`/Last-Event-ID replay, single terminal emitter, Studio
  toasts/activity, legacy mapping, recovery and cancellation.
- [systems/prompt-enhancement](systems/prompt-enhancement.md) — the `llm.py`
  dispatcher: Gemma built-in vs Ollama, keep_alive unload discipline,
  vision-model badges, VLM structured extraction.
- [systems/engineering-audit-2026-11](systems/engineering-audit-2026-11.md) —
  code-level audit and remediation ledger: original P0 security/data/render
  blockers, evidence with file:line, what works, remaining product gaps, and the
  resolved doc contradictions.
- [systems/studio-shell](systems/studio-shell.md) — the implemented Studio shell:
  Story/Generate/Edit/Review/Library/Deliver/Shorts/Setup surfaces, board + timeline
  as peers, canonical shot graph, separate stores, canvas/timeline engines,
  performance budgets, and the remaining scoped gaps.
- [systems/studio-ux-audit-2026-11](systems/studio-ux-audit-2026-11.md) — 2026-11
  chrome/density audit: measured 1024px nav collision, board-toolbar overflow,
  clipped rail labels, hidden Inspector tabs, typography/control-size evidence,
  missing UX functionality, and the completed F6 stabilization pass.
- [systems/studio-media-audit-2026-11](systems/studio-media-audit-2026-11.md) —
  2026-11 media-open audit: passive media, missing preview paths, legacy localhost
  URLs, caption misclassification, and the global image/video/audio/text viewer.
- [systems/frontend-ux-audit-2026-11-18](systems/frontend-ux-audit-2026-11-18.md) —
  2026-11-18 whole-frontend UX audit: eight surfaces at 1440×900 and 390×844,
  axe-clean but 24 findings (`F7-01…F7-24`) covering untruthful save/shot status,
  project-global Library media, 680px of Generate rails, a 124px mobile Story
  board, truncated safety warnings, timeline overflow, and copy/tone semantics.
  Resolution shipped 2026-11-18 across all seven phases; plan:
  [[roadmap/frontend-ux-plan-2026-11-18]].
- [systems/f7-caveats-audit-2026-11-18](systems/f7-caveats-audit-2026-11-18.md) —
  caveat audit behind F7: **LUT preview blocked by cross-origin taint**, relink
  leaves take/job provenance and old files stale, save conflict/undo truth gaps,
  GET→PUT 500 on `created_at`, 121px mobile Edit timeline, drawer semantics, and
  18 findings (`F8-01…F8-18`). **All F8 findings are fixed**; resolution evidence
  is on the audit page.
- [systems/memory-management-audit-2026-11-18](systems/memory-management-audit-2026-11-18.md) —
  2026-11 memory audit: live M3 Max snapshot, MPS/MLX allocator research, agentic
  memory/context research, and 20 findings (`MEM-01…MEM-20`). Headline: the
  reported ceiling is **17.3 GB above the real process cap**, active jobs are
  observed but not guarded, and agentic memory is effectively absent.
- [systems/performance-audit-2026-11-18](systems/performance-audit-2026-11-18.md) —
  2026-11 performance audit: local M3 Max benchmarks, 2026 research on LTX/MLX,
  MPS, VideoToolbox, SQLite/WAL, React/WebCodecs, and 22 findings
  (`PERF-01…PERF-22`). Headline: a 40-step CFG schedule drives a distilled
  checkpoint; the MPS loader still uses slow mmap; VideoToolbox is unused.
- [systems/f9-caveats-audit-2026-11-18](systems/f9-caveats-audit-2026-11-18.md) —
  caveat audit behind F8: **stale Running cards after job disappearance**, duplicate
  `/active_jobs` pollers, silent LUT source failures, orphaned relink thumbnails,
  legacy-path relink URLs, repair/delete lineage state, LUT compare/precision,
  mobile transport, and 21 findings (`F9-01…F9-21`). **All F9 findings are fixed**;
  resolution evidence is on the audit page.
- [systems/llm-settings-audit-2026-11](systems/llm-settings-audit-2026-11.md) —
  LLM configuration audit **and implementation record**: Kubona reference patterns,
  the old Gemma/Ollama gaps, the shipped L0–L3 plane (persisted providers/roles,
  discovery, verify, sealed keys, air-gap, Studio panel), and the deferred L4 work.
- [systems/memory-management-audit-2026-11](systems/memory-management-audit-2026-11.md) —
  memory/OOM audit: the 128 GB incident, measured model and external-engine costs
  (Strata 33 GB, oMLX 6.5 GB), static-estimate failure, gap register M-01…M-29,
  engine adapters, budget/lease/eviction design, phases M0–M5, **and the full
  implementation record** (priority/same-slot queue, memory waits, downgrade
  ladder, metrics, tiled/buffer release, SAM lazy load, benchmark tool) with live
  refusal/cancel/downgrade evidence.
- [systems/job-notifications-audit-2026-11](systems/job-notifications-audit-2026-11.md) —
  generation notification audit: the missing Studio toast layer and SSE consumer,
  live reproduction, gap register N-01…N-28, event catalog, the N0–N5 plan, **and
  the full implementation record**: typed events, replay, single terminal, toasts,
  activity center, inline progress, resync/watchdog, metrics, history retry, ETA,
  desktop notifications, dock badge, webhook, and per-project mute, with live
  probes.
- [systems/github-issues-triage-2026-11](systems/github-issues-triage-2026-11.md) —
  triage of the five open GitHub issues against the November implementation:
  the fixed AE download, CUDA VRAM admission/OOM mitigation, API documentation,
  and the still-open Wan 2.2 / quantized-checkpoint feature work.
- [systems/llm-local-first-next-2026-11](systems/llm-local-first-next-2026-11.md) —
  local-first LLM audit **and implementation record**: live measurements from
  Ollama/oMLX/Strata, findings F1–F11, phases L5–L9 (instances, capability truth,
  warm-up/streaming, eval, memory discipline, cloud budget/profiles), and the
  remaining caveats.
- [systems/comfyui-bridge](systems/comfyui-bridge.md) — the generic ComfyUI
  bridge: API-format workflow submission, websocket/polling progress, output
  streaming into assets and shots, declared external residency, port rules, and
  the model-free smoke test.
- [systems/comfyui-frontend-audit-2026-11](systems/comfyui-frontend-audit-2026-11.md) —
  the 2026-11 ComfyUI frontend audit: capability matrix, C-01…C-24 gap register,
  live validation evidence, and recommended bindings/media/library/results work.

## Research
- [research/model-landscape-2026-11](research/model-landscape-2026-11.md) —
  November 2026 model/workflow scan: LTX-2.5, Wan 2.2, HunyuanVideo 1.5,
  MiniMax H3, Qwen-Image 2.1, Qwen3-TTS, SeedVR2, ComfyUI, and OTIO; licenses,
  VRAM floors, and the decisions that re-sequence the roadmap.

## Models
- [models/ltx-2](models/ltx-2.md) — LTX-2 19B dual-stream transformer: the three
  pipelines (ti2vid, ic_lora, keyframe), two-stage generation, checkpoints,
  pipeline internals and gotchas.
- [models/ltx-2-5](models/ltx-2-5.md) — LTX-2.5 22B: gated diffusers adapter, native
  synchronized audio, CPU offload/VAE tiling, and the `ltx25` pipeline.
- [models/flux-2](models/flux-2.md) — Flux 2 Klein 9B: image synthesis, IP-Adapter
  reference conditioning, RePaint inpainting, True CFG, AE hot-swap, init/loading
  and denoise-loop internals.
- [models/sam-3](models/sam-3.md) — SAM 3 microservice (port 8002): text/box/click
  segmentation, detection, bidirectional video object tracking; startup sequence,
  predictor internals, proxy routes, MPS port notes.

## Pipelines
- [pipelines/chained-generation](pipelines/chained-generation.md) — Quantum
  Alignment ("Smart Continue"): autoregressive chunk-by-chunk video past the
  505-frame limit, with exact latent-grid overlap math.
- [pipelines/inpainting](pipelines/inpainting.md) — SAM mask → Flux RePaint loop:
  mask acquisition paths, the noisy-known blend, job persistence, save paths.

## Concepts
- [concepts/memory-slots](concepts/memory-slots.md) — the measured budget and
  residency slots: admission, leases, TTL eviction, engine adapters, pressure/swap
  policy, and what each heavy model really costs.
- [concepts/mps-first](concepts/mps-first.md) — Apple Silicon as a first-class
  citizen: the VAE-decode, dtype, and memory fixes that keep MPS from shipping
  black frames.
- [concepts/story-elements](concepts/story-elements.md) — Characters/Locations/
  Objects with `@trigger` words: ElementManager injection, IP-Adapter identity,
  the 8-signal ElementMatcher.
- [concepts/project-history](concepts/project-history.md) — build timeline
  distilled from the dev_diary: milestones, the MPS campaigns, the acquisition
  announcement, open threads.

## Workflows
- [workflows/getting-started](workflows/getting-started.md) — from zero to studio:
  prerequisites, model downloads, the three services, first generation.
- [workflows/script-to-video](workflows/script-to-video.md) — end-to-end:
  paste script → parse → match elements → thumbnails/pre-viz → generate shots →
  timeline assembly.
- [workflows/masking-tracking](workflows/masking-tracking.md) — selective
  editing: MaskingCanvas 4-mode masking → inpaint; TrackingPanel session →
  propagate → export masks.

## Decisions
- [decisions/sam3-microservice](decisions/sam3-microservice.md) — why SAM 3 runs
  as an isolated microservice on port 8002 instead of inside the backend.
- [decisions/cloud-engines](decisions/cloud-engines.md) — PROPOSAL: optional
  per-engine cloud fallbacks (Veo/Imagen/Gemini) that bypass the local GPU path;
  not implemented.

## Research
- [research/frontend-video-platforms-2026](research/frontend-video-platforms-2026.md) —
  Nov 2026 UI/UX survey of Runway, Google Flow, Kling, Luma, Pika, LTX Studio,
  Adobe, Descript, CapCut, OpusClip, Boords, Shot Designer, Frame.io, Figma Weave,
  Krea, ComfyUI, tldraw, Milanote, Kubona; canvas/timeline/agent patterns.
- [research/video-models-2026](research/video-models-2026.md) — open video-model
  landscape since LTX-2 (Oct 2026): LTX-2.5, Wan 2.2/3.0, MiniMax H3, Kandinsky 6.0,
  licensing traps, replace-or-complement verdict.
- [research/image-models-identity-2026](research/image-models-identity-2026.md) —
  open image models + identity tech (Oct 2026): Z-Image Turbo, FLUX.2 family,
  Qwen-Image line, native multi-ref conditioning, LoRA identity.
- [research/world-models-3d-camera-2026](research/world-models-3d-camera-2026.md) —
  world models, 3D generation, camera control, depth, segmentation/tracking,
  video inpainting (Oct 2026); keep-SAM-3 verdict.
- [research/inference-infra-2026](research/inference-infra-2026.md) — inference
  engines, quantization, Apple Silicon/MLX/MPS, VRAM management (Oct 2026);
  slot-architecture verdict: keep.
- [research/frontend-ux-workflows-2026](research/frontend-ux-workflows-2026.md) —
  React 19, canvas timelines, Tauri 2, copilot UX, agentic video workflows,
  2026 local LLMs, eval harnesses (Oct 2026).

## Roadmap
- [roadmap/v2-roadmap](roadmap/v2-roadmap.md) — the v2 design roadmap: guiding
  principles, per-slot model recommendations, architecture deltas, UI/UX direction,
  phased plan (Phase 0 verify → Phase 1 models → Phase 2 architecture → Phase 3
  UX), risks, and open questions for Mainza. Phase 1+ now gated by the 2026-11 audit.
- [roadmap/open-work-resolution-2026-11](roadmap/open-work-resolution-2026-11.md) —
  the current open-work inventory and P0–P3 resolution plan: red CI dependency
  pin, PR #7/#8, issue triage, CUDA/quantized work, sidecar packaging, legacy
  cutover, and product gaps with acceptance criteria.
- [roadmap/comfyui-frontend-plan-2026-11](roadmap/comfyui-frontend-plan-2026-11.md) —
  the phased ComfyUI frontend plan (C0 trust → C1 binding/media → C2 library →
  C3 results/batch → C4 remote → C5 polish), with acceptance criteria.
- [roadmap/frontend-ux-plan-2026-11-18](roadmap/frontend-ux-plan-2026-11-18.md) —
  the F7 whole-frontend UX plan: F7-0 truthful status → F7-1 project media →
  F7-2 one rail per surface → F7-3 mobile shell → F7-4 timeline/preview parity →
  F7-5 language/tone → F7-6 verification gate. Maps every finding to a phase,
  with acceptance criteria and effort. **Implemented 2026-11-18; preview caveat
  corrected by F8.**
- [roadmap/f7-caveats-plan-2026-11-18](roadmap/f7-caveats-plan-2026-11-18.md) —
  the F8 caveat plan: F8-0 contract round-trip → F8-1 save/shot truth → F8-2 LUT
  preview truth → F8-3 relink/repair provenance → F8-4 mobile Edit/drawer →
  F8-5 verification. **Implemented 2026-11-18; caveats re-audited in F9.**
- [roadmap/memory-management-plan-2026-11-18](roadmap/memory-management-plan-2026-11-18.md) —
  the M0–M5 memory plan: truth/allocator stats → calibration/loader/residency/MLX
  → runtime guard/children → browser/document → agentic memory/context → eval and
  security. Extends performance P2. **Planned, not started.**
- [roadmap/performance-plan-2026-11-18](roadmap/performance-plan-2026-11-18.md) —
  the P0–P6 performance plan: measurement lab → distilled schedule/loader/prompt
  cache → memory/VAE/MLX/machine tiers → VideoToolbox/proxies → frontend runtime
  → SQLite/API → CI gates. **Planned, not started.**
- [roadmap/f9-caveats-plan-2026-11-18](roadmap/f9-caveats-plan-2026-11-18.md) —
  the F9 caveat plan: F9-0 job truth → F9-1 relink/repair safety → F9-2 preview
  truth → F9-3 save/delete safety → F9-4 mobile transport/dialogs → F9-5 gate.
  **Implemented 2026-11-18.**
- [roadmap/implementation-plan-2026-11](roadmap/implementation-plan-2026-11.md) —
  the sequenced fix-and-ship plan. Scoped P0–P5, F0–F5, and F6 are complete; the page
  lists remaining/deferred work (bundled sidecar, legacy cutover, Phase 6 v2 lane,
  scoped v1 gaps) and the verification matrix.
- [roadmap/frontend-refactor-2026-11](roadmap/frontend-refactor-2026-11.md) — the
  strangler rebuild of the studio: F0 shell/tokens/contracts → F1 story/board →
  F2 edit/timeline → F3 generate/takes → F4 agent/review/library/deliver → F5
  shorts/lenses. Includes gates, effort, risks, decisions.

## Sources (read-only, in docs/, dev_diary/, mv_plan/)
- `docs/01_system_architecture.md` — component diagrams and subsystem interactions
- `docs/02_data_models.md` — ER diagrams, Pydantic schemas, TypeScript types
- `docs/03_ai_pipelines.md` — LTX-2, Flux 2, SAM 3 pipeline analysis with flow diagrams
- `docs/04_frontend_state.md` — Zustand store architecture and optimization strategies
- `docs/05_execution_flow.md` — sequence diagrams for all generation workflows
- `docs/06_file_dependency.md` — import graphs and module analysis
- `docs/flux2-bible.md` — deep-dive into Flux 2 integration
- `docs/ltx2-bible.md` — deep-dive into LTX-2 integration
- `docs/sam3-bible.md` — deep-dive into the SAM 3 microservice
- `docs/CODEBASE_ANALYSIS.md` — codebase analysis
- `docs/user_guide.md` — the definitive Studio-first user guide
- `skills/skills/milimo-frontend-state-manager/SKILL.md` — frontend state conventions
- `dev_diary/` — build log: social updates, blog posts, explainer notes
- `mv_plan/google_cloud_integration.md` — cloud-engines proposal (plan, not implemented)
