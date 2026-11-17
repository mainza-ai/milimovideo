---
type: overview
created: 2026-10-09
updated: 2026-11-17
tags: overview local-first nle open-source studio
sources:
  - README.md
  - ../docs/user_guide.md
related:
  - systems/architecture.md
  - systems/studio-shell.md
  - roadmap/implementation-plan-2026-11.md
  - workflows/getting-started.md
---

# Overview: Milimo Video

**Milimo Video** is an open-source, AI-native cinematic studio — a professional
**non-linear editor (NLE)** for filmmakers, running entirely **local-first** on the
filmmaker's own machine. Apache 2.0 licensed. Author: Mainza Kangombe.

It is one application with two front ends during the strangler cutover:

- **Studio** at `/studio` — Story, Generate, Edit, Review, Library, Deliver, Shorts,
  and Setup. This is the current shell.
- **Legacy NLE** at `/` — frozen fallback for the two-clean-release window. It can
  be forced back onto `/studio` with `VITE_STUDIO_ENABLED=false`.

The Tauri desktop shell hosts the same Studio UI with native Open/Save dialogs and
a backend supervisor.

## Models

| Model | Role |
|---|---|
| **LTX-2 19B** | Dual-stream transformer for cinematic video generation (text-to-video, image-to-video, keyframe interpolation) |
| **Flux 2 Klein 9B** | High-fidelity image synthesis with IP-Adapter reference conditioning and RePaint inpainting |
| **SAM 3** | Text-prompted segmentation, click-to-segment, and video object tracking (standalone microservice) |
| **Gemma 3** | Prompt enhancement and narrative direction |
| **Depth Anything V2 Small** | Optional local monocular depth for the Edit overlay |

## What makes it a studio, not a demo

- **One project graph, three lenses.** Story (script/board/shot list), Generate
  (shot workspace/takes/ingredients), and Edit (stage/timeline) all read the same
  `Shot` graph. Board position never defines sequence order.
- **Immutable takes.** Every completed generation is a take with seed, model, and
  lineage. Selecting a take repoints the shot; old takes remain comparable.
- **Truthful render.** The export is the computed cut: magnetic V1 from
  `sequence_index`, free V2/A1, alpha compositing, embedded audio + A1 mix, trims,
  speed, fades, keyframes, captions, render range, and persisted encoding settings.
- **A deterministic default.** Seed lock is on; the same prompt and parameters
  reproduce the same take. Randomizing a new take is explicit.
- **Plan before spend.** The command dock proposes a diff with model/time/VRAM
  estimates; approval is required and applying a plan is one undo step.
- **Durable local jobs.** A `QueuedJob` ledger, one global GPU semaphore, one OOM
  retry, startup recovery, pause/resume dispatch, cancellation, and SSE progress.
- **Local-first security.** Loopback default, CORS allowlist, optional bearer token,
  sanitized uploads, safe media Range serving, and no cloud dependency in the hot
  path.

## Current State (2026-11-17)

The scoped P0–P5, F0–F5, and F6 chrome-stabilization work is complete:

- security, canonical timeline/persistence, render fidelity, scheduler, model truth,
  UX trust, tests/CI;
- Studio shell and all surfaces;
- review/share/agent, library/deliver, shorts/lenses, Tauri shell, queue pause,
  seed lock, encoding controls, integrity scan/repair, and native open/save.

Verification: **84 backend tests, 56 Vitest tests, 17 Playwright tests**, clean
ruff/tsc/ESLint, build + bundle budget, `cargo check`, and `verify_setup` with zero
failures.

Remaining/deferred work is explicit in
[[roadmap/implementation-plan-2026-11.md]]: bundled Python sidecar runtime, legacy
cutover, Phase 6 v2 model/eval lane, and scoped-v1 gaps (transitions/LUT/titles/A2,
embedding search, shorts/lens template packs). Visual baselines are local and
platform-tolerant.

## Related

- [[systems/architecture.md]] — how the pieces fit together
- [[systems/studio-shell.md]] — the implemented Studio shell
- [[systems/data-models.md]] — persistence and migrations
- [[roadmap/implementation-plan-2026-11.md]] — status and remaining work
- [[workflows/getting-started.md]] — from zero to studio
