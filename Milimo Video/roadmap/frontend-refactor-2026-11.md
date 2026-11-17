---
type: roadmap
created: 2026-11-16
updated: 2026-11-16
tags: frontend refactor ux canvas timeline studio shell plan
sources:
  - wiki/research/frontend-video-platforms-2026.md
  - wiki/systems/studio-shell.md
  - wiki/systems/engineering-audit-2026-11.md
  - https://github.com/mainza-ai/kubona
related:
  - ../systems/studio-shell.md
  - ../research/frontend-video-platforms-2026.md
  - ../systems/engineering-audit-2026-11.md
  - implementation-plan-2026-11.md
  - v2-roadmap.md
---

# Front-End Refactor — November 2026

## Purpose

Rebuild Milimo Video's front end as a coherent local-first studio:

> **One project graph. Three synchronized lenses — Story, Generate, Edit — plus
> Review, Library, and Deliver. A board for meaning, a timeline for time, and an
> agent that proposes before it spends.**

This is not a reskin. It replaces the current 7-slice God store, tab-switched
layout, hardcoded API calls, destructive generation flow, and disconnected
storyboard/timeline models. It is a **strangler refactor**: the new shell ships
behind a route/flag, surfaces migrate one at a time, and the old shell is removed
only after parity and two clean releases.

The design evidence is in
[[../research/frontend-video-platforms-2026.md]]; the target architecture is in
[[../systems/studio-shell.md]]; the correctness gates are in
[[implementation-plan-2026-11.md]].

## Execution Status — 2026-11-16

The shell is now live at `/studio` behind a lazy route; the legacy editor remains
at `/`.

| Stage | Status | What exists |
|---|---|---|
| **F0 Foundation** | **Started** | `studio.css` tokens, shell, mode rail, top bar, inspector, command dock, status strip, `editorStore` (persisted camera/layout), API base/token, `/studio` lazy route, component mount test, Setup surface with real device/slot/pipeline/model/disk state and explicit activation |
| **F1 Story** | **Partial** | Story spine with master-reference filter, board canvas (pan/zoom/drag/LOD/minimap), areas with inline rename/resize, typed links (thread/blocking/reference), list view, script view, selection sync, viewport-culled cards for large boards, board layout persisted with revision/CAS + separate view save |
| **F2 Edit** | **Partial** | Stage monitor, legacy playback clock, V1/V2/A1 clips, playhead, seeking, zoom; tested frame-accurate trim (in/out), slip, roll, blade at playhead, and ripple delete through pure ops; per-clip audio gain/mute persists and reaches the render filtergraph; caption cues persist in C1, edit in-place, and burn in through generated ASS; no keyframe UI yet |
| **F3 Generate** | **Partial** | Shot workspace, prompt/params, generate/cancel, immutable take history from completed jobs with select-take |
| **F4 Review** | **Partial** | Version stack reads the same take history; `POST /projects/{id}/plan` returns deterministic plan cards (diff, estimate, warnings) applied through the undoable store; frame-accurate comments with resolve/delete and per-shot approvals persist via `/projects/{id}/review`; `@Agent` comments route into the plan dock for approval; plan patches are one undo step via top-bar/⌘Z undo-redo; a live “needs you” pill counts unresolved comments and change requests; share links/activity stay open |
| **F4 Library/Deliver** | **Partial** | Library assets/references/complete-take lineage with select; Deliver presets 16:9/9:16/1:1/4:5, render history, caption cue count; semantic search and share links open |
| **F5 Shorts/lenses** | **Not started** | — |

**Next:** share/“needs you” review links, then move document state off the
legacy store. Do not cut over `/` until F2 review/render parity.

## Why This Refactor

The audit found the current UI cannot be trusted with a real project:

- V1 order and conditioning do not round-trip.
- Export does not represent the cut (timing, V2, audio).
- Regeneration overwrites takes; there is no review/version surface.
- Image/inpaint jobs bypass the queue and freeze the app.
- 74 hardcoded localhost URLs; contracts drift between snake_case and camelCase.
- The store persists transient state, pollutes undo, and has no save conflict
  model.
- There is no canvas, no command palette, no plan/approval agent, and no
  accessible list equivalent for spatial editing.

A better model picker cannot fix those. The shell, state model, and data contracts
must change together.

## Architecture Deltas

| Area | Current | Target |
|---|---|---|
| Layout | Tab-switched center; timeline always bottom; modals for projects/exports | Mode shell with Story/Generate/Edit/Review/Library/Deliver; resizable panels; command dock |
| Spatial planning | Storyboard grid only | Infinite board with areas, links, pins, LOD, minimap, list view |
| Timeline | DOM-only, 3 tracks, no embedded audio/captions | DOM handles + canvas media, V2/V1/A1/A2/captions, keyframes, transitions |
| State | 7-slice God store; whole-project persist + zundo | Document/editor/job/agent stores; Query for server state; patch undo |
| Contracts | Handwritten snake/camel mapping | Generated OpenAPI types + Zod; one API client |
| Generation | Destructive one-shot; modal; no takes | Shot workspace; ingredients; immutable takes; compare; queue |
| Agent | None | Command dock with context chips, plan, cost/time, approval, undoable apply |
| Review | None | Version stacks, compare, timecoded comments, approvals, activity |
| Assets | Flat gallery | Library with master references, semantic search, lineage, collections |
| Performance | Per-frame project subscribers; full timeline DOM | Culling/LOD; isolated playhead; virtualized lists; canvas media layer |

## Principles

1. **Correctness over chrome.** The new shell consumes the canonical backend
   model from the audit plan; it does not invent a second one.
2. **Board and timeline are peers.** Neither is a modal or a hidden tab. Both read
   the same `Shot`/`Take` graph.
3. **Shot cards are the unit of work.** Card ↔ row ↔ clip are one object.
4. **Takes are immutable.** No destructive regeneration.
5. **Agent shows the plan.** Model, objects, time/VRAM, diff, approval.
6. **Local-first is visible.** Device, slots, queue, disk, and save state are
   always legible.
7. **Accessible by construction.** Every canvas has a list view; every action has
   a keyboard path.
8. **Measure it.** 60fps budgets, bundle budgets, and interaction latency are
   acceptance criteria, not aspirations.

## Target Outcomes

- **Story** — script, board, and shot list are three views of one project.
- **Generate** — shot workspace with ingredients, references, takes, compare, and
  a queue that never runs two heavy models at once.
- **Edit** — a frame-accurate timeline with V1/V2/A1/A2/captions, embedded audio,
  transitions, and render presets.
- **Review** — version stacks, compare viewer, timecoded comments, approvals.
- **Library** — assets, master references, takes, models, semantic search.
- **Deliver** — film and shorts presets, variants, captions, exports.
- **Agent** — context-aware command dock with plan-before-spend and undoable
  application.
- **No hardcoded localhost; no God store; no duplicate contracts.**

## New Workflows Enabled

1. **Spatial story development** — write, board, and shot-list a film in one
   place; drag shots into sequence; see continuity and references around each
   scene.
2. **Shot-level generation with takes** — define ingredients, preview a frame,
   generate, compare takes, select, and keep every version.
3. **Transcript-first rough cut** — edit dialogue-driven footage by editing text;
   captions and cuts follow.
4. **Review and approval** — share a cut, comment at frame 42, compare versions,
   approve; `@Agent` can propose a fix.
5. **Shorts repurposing** — one long cut → scored candidates → reframed/captioned
   variants for 9:16, 1:1, 4:5, 16:9.
6. **Blocking and previz** (later) — top-down camera diagram and 3D stage that
   feed camera/motion prompts.
7. **Reusable skills** — save a successful plan as a local recipe with typed
   inputs and re-run it on a new selection.

## Migration Strategy

### Strangler, not big bang

- Keep the current app at `/` while the new shell lives at `/studio`.
- New code under `web-app/src/studio/` and `web-app/src/design-system/`; freeze
  legacy `components/` except for P0 fixes.
- Feature flags per surface: `studio.shell`, `studio.board`, `studio.edit`,
  `studio.generate`, `studio.review`, `studio.library`, `studio.deliver`.
- Migrate in this order: shell → story/board → edit/timeline → generate/takes →
  review/library → deliver/shorts.
- Cutover when: save/reload is lossless, export matches the cut, and two releases
  have no P0 regressions.
- Delete the legacy shell only after cutover; keep a one-release rollback flag.

### Contract-first

The front end starts with generated types and a single API client. If the backend
contract is not ready (audit Phase 1/G1), the shell can still be built against
fixtures, but no surface ships until G1 is green.

### Data migration

- Backfill `sequence_index` and board layout on first open.
- Preserve all existing shots/takes/assets; no destructive rewrite.
- Keep old URLs readable; new URLs are canonical.
- Provide a “Repair project” action that reports orphans and missing media.

## Phases

Effort scale: **S** < 1 week, **M** 1–2 weeks, **L** 2–4 weeks, **XL** > 4 weeks.
Assumes two front-end engineers + part-time design.

### F0 — Foundation (2–3 weeks)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| F0-01 | Design tokens + primitives: color, type, spacing, radius, motion, focus, density | Dark/high-contrast pass; no ad-hoc hex in new code | M |
| F0-02 | App shell: top bar, mode rail, resizable inspector, status strip, command dock container | Layouts persist; responsive at 1440/1024/768 | M |
| F0-03 | API client + SSE client + generated types + Zod validation | No hardcoded localhost; contract test fails on drift | M |
| F0-04 | State skeleton: document/editor/job/agent stores; TanStack Query | Selection/playback isolated; no broad project subscribers | M |
| F0-05 | Routing + feature flags + legacy fallback | `/studio` loads without touching legacy; rollback works | S |
| F0-06 | Test harness: Vitest, Playwright, axe, visual regression | One smoke test per surface; CI job wired | M |
| F0-07 | Accessibility baseline: skip link, focus rings, keyboard map, reduced motion | axe 0 critical on shell; keyboard-only navigation | S |
| F0-08 | Import Kubona patterns (Apache 2.0) with attribution: camera, LOD, minimap, list view helpers | Attribution file; no quantum domain code | S |

**Gate F0:** shell renders, navigates, persists layout, passes axe, and has a
working API client with fixtures.

### F1 — Story: Script, Board, Shot List (3–4 weeks)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| F1-01 | Canonical `Shot`/`Scene`/`Take` types and document store | Save/reload diff is empty | M |
| F1-02 | Story spine (Acts/Scenes/Shots) + filters/search | 500 shots scroll smoothly | M |
| F1-03 | Board canvas: camera, pan/zoom, marquee, move/resize, snap, align/distribute | 500 cards at 60fps pan/zoom | L |
| F1-04 | Cards: scene, shot, take, element, reference, note, task | Summary/compact/detail LOD; hover actions | L |
| F1-05 | Areas (Act/Scene/Look/Master References) + collapse + color | Drag children with area; persist | M |
| F1-06 | Typed links + edge visibility + create/remove | Links survive reload; list view shows them | M |
| F1-07 | Minimap, fit, focus, follow-live, pin | Minimap pans; pinned cards never auto-move | M |
| F1-08 | Accessible list/table equivalent | Every canvas action available in list view | M |
| F1-09 | Script view + shot list table with inline edit/bulk actions | Text edit syncs to shot; table sorts/filters | L |
| F1-10 | Selection sync: board ↔ shot list ↔ inspector ↔ timeline | Selecting one selects all; scroll-to-selection | M |
| F1-11 | Board layout persistence with revision/CAS + conflict UI | Two tabs cannot silently overwrite | M |
| F1-12 | Undo for board ops, separate from document undo | Move/resize/link/group undo correctly | M |

**Gate F1:** create a 10-shot project, rearrange on the board, edit in the shot
list, save, hard reload — order, cards, areas, links, and selection are identical.

### F2 — Edit: Stage, Timeline, Inspector (3–4 weeks)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| F2-01 | Program monitor + shot monitor + compare modes | A/B wipe and side-by-side at frame accuracy | M |
| F2-02 | Stage overlays: safe area, grid, mask, tracking, depth, camera path | Toggleable; no layout shift | M |
| F2-03 | Timeline engine: camera, ruler, tracks, clips, playhead | 200 clips at 60fps scrub | L |
| F2-04 | Magnetic V1 from `order_index`; free V2/A1/A2 | Reorder persists; no `start_frame` for V1 | M |
| F2-05 | Tools: select, blade, ripple, roll, slip, slide, trim, rate | Keyboard + pointer parity | L |
| F2-06 | Audio: waveform, gain, fades, mute/solo, embedded V1/V2 audio | Generated audio audible and exportable | L |
| F2-07 | Captions track: transcript import, SRT/VTT, burn-in option | Captions follow cuts and export | M |
| F2-08 | Keyframes: opacity/scale/position/volume | Keyframes survive save/reload | L |
| F2-09 | Inspector tabs: Clip/Take/Audio/Effects/History | Contextual, resizable, keyboard accessible | M |
| F2-10 | Render range, presets, progress, cancel | Export matches the cut; progress granular | M |
| F2-11 | Playback clock isolated; no broad re-renders | No React re-render storm at 60fps | M |
| F2-12 | Timeline list/table equivalent + keyboard edit commands | Full edit path without pointer | M |

**Gate F2:** a 3-shot cut with V1, V2 alpha overlay, generated audio, music,
captions, and a transition exports correctly; reload preserves every trim,
placement, and keyframe.

### F3 — Generate: Shot Workspace & Takes (2–3 weeks)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| F3-01 | Shot workspace: list + stage + takes grid + ingredients | Generate a shot without leaving the surface | M |
| F3-02 | Ingredient composer: @elements, references, start/end frames, voice, style | Drag/drop and @-mention; context chips | L |
| F3-03 | Params with progressive disclosure: model, duration, resolution, seed lock, guidance, pipeline | Default 5–6 controls; advanced collapsed | M |
| F3-04 | Take model: immutable takes, select/restore/favorite/compare | Regenerate never overwrites a take | L |
| F3-05 | Queue: estimate time/VRAM, position, cancel, retry, batch | Batch 10 shots; one heavy model at a time | L |
| F3-06 | Preview-first: concept frame/image before video; dry run | Preview before spend | M |
| F3-07 | Inpaint/edit-frame flow: mask, track, prompt, preview, apply as new take | Result attached to shot as a take | M |
| F3-08 | Batch generation matrix: status per shot, pause/resume/cancel | No duplicate jobs; no orphan state | M |

**Gate F3:** generate 10 shots with references; cancel one, retry one, compare
takes, select a take, and reload — all state correct, no duplicate jobs.

### F4 — Agent, Review, Library, Deliver (3–4 weeks)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| F4-01 | Command dock: Ask/Direct/Generate/Find modes + context chips | Agent sees selection/board/time range | L |
| F4-02 | Plan cards: steps, models, affected objects, time/VRAM, diff preview | Ask-before-spend default; approve/edit/cancel | L |
| F4-03 | Apply plans as undoable patches | One undo restores pre-plan state | M |
| F4-04 | Skills/recipes: save/re-run a plan with typed inputs | Re-run on a new selection | M |
| F4-05 | Review: version stacks, compare, comments, approvals, activity | Comment at frame; approve; reload persists | L |
| F4-06 | Share link + “Needs you” tray | Reviewer can comment/approve without the editor | M |
| F4-07 | Library: assets, master references, takes, models; semantic search | Find “neon city rain” locally | L |
| F4-08 | Deliver: film/shorts presets, variants, caption burn-in, render queue | 16:9 + 9:16 + 1:1 + 4:5 from one cut | L |
| F4-09 | Model library refresh: real activation, VRAM, LoRA stack, queue | “Activate” actually loads; status truthful | M |
| F4-10 | Agent-in-comments with approval | `@Agent` proposes, never silently applies | M |

**Gate F4:** a reviewer comments on frame 42, the editor applies the fix through
the agent plan, the new take is approved, and the old version remains comparable.

### F5 — Shorts, Lenses, Packaging (4–6 weeks, optional/stretch)

| Task | Change | Acceptance | Effort |
|---|---|---|---|
| F5-01 | Shorts workspace: source/transcript → candidates → scores → reframe → captions → variants | One 20-min source → 5 publishable vertical clips | L |
| F5-02 | Pipeline lens: node graph for a shot recipe, plan/validate/cost, app mode | Reproducible recipe; downstream-only rerun | XL |
| F5-03 | Blocking lens: top-down camera/character diagram, shot list sync, animation | Camera moves update shot list | L |
| F5-04 | Previz 3D: three.js stage, camera path, depth snapshot, lazy/fail-closed | 3D loads only on demand; no blank canvas | XL |
| F5-05 | Tauri 2 sidecar packaging | Native file open/save; same shell | L |
| F5-06 | Collaboration spike: Yjs/CRDT for board/comments | Two clients edit without conflict | XL |

Do not start F5 before F2/F4 gates. The pipeline/blocking/3D lenses are
differentiators, not v1 requirements.

## Gates

| Gate | Criteria |
|---|---|
| **F0** | Shell + tokens + API client + tests + axe; `/studio` loads behind flag |
| **F1** | 10-shot story round-trip; 500 cards at 60fps; list view parity |
| **F2** | Export matches cut: V1/V2/audio/captions/keyframes; 200 clips at 60fps |
| **F3** | Takes are immutable; batch queue; cancel/retry; no duplicate jobs |
| **F4** | Agent plan/approval; review/version/comments; library/deliver |
| **F5** | Shorts pipeline; optional lenses/packaging |

Front-end gates depend on backend gates:

- F1 needs **G1** (canonical timeline/persistence).
- F2 needs **G2** (render fidelity).
- F3 needs **G3** (unified GPU queue).
- F4 needs review/agent APIs (new, small).

## Team & Effort

- **Two front-end engineers + part-time design:** F0–F4 in **12–14 weeks**.
- **One front-end engineer:** **20–24 weeks**; not recommended alongside backend
  correctness work.
- **Design:** 2–3 weeks for tokens, shell, board/card system, timeline, review.
- **QA/automation:** 1 engineer part-time from F0; full-time at F2/F4.

Parallelization:

- F0 starts immediately.
- F1 and F2 can overlap once G1 lands (different engineers, shared canvas/timeline
  primitives).
- F3 starts when G3 is green; F4 can start on fixtures and integrate last.
- F5 only after F4.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Big-bang rewrite stalls | Strangler route/flag; ship shell first; legacy fallback |
| Dual maintenance | Freeze legacy except P0; one release max |
| Canvas performance | Budget tests at 500 cards; culling/LOD from day one; no per-frame React |
| Timeline rewrite destabilizes editing | Build timeline engine behind flag; golden render/trim tests |
| Contract drift | Generated types + Zod + contract tests; no `List[dict]` |
| Agent scope creep | Plan/approval first; execution in F4; no autonomous spend |
| Design overreach | 5–6 default controls; progressive disclosure; no node graph home |
| Accessibility regression | List view + axe + keyboard tests per gate |
| Tauri/distraction | Defer to F5; browser shell first |

## Decisions Needed

1. **Canvas engine:** custom DOM transform (Kubona-style) vs tldraw SDK vs React
   Flow. **Recommendation:** custom DOM for Story Board (rich cards, control,
   license); React Flow only for the Pipeline lens.
2. **Framework:** keep Vite + React 19 (recommended) vs migrate to Next.js. Do not
   add SSR to a local-first studio.
3. **State:** Zustand + TanStack Query + Immer patches (recommended) vs a single
   new store.
4. **Workspace presets:** Film/Shorts as project presets (recommended) vs separate
   apps.
5. **Review sharing:** local link + token (recommended) vs cloud.
6. **Tauri:** after F4 (recommended) vs earlier.
7. **3D:** after F2/F4 (recommended) vs in v1.

## Definition of Done

- `/studio` is the default shell; legacy is removed after two clean releases.
- A 10-shot film can be written, boarded, generated, cut, reviewed, and exported
  without leaving the shell.
- Save/reload is lossless for order, trims, placement, conditioning, takes,
  comments, and board layout.
- Export matches the cut: timing, V2 alpha, V1/V2 audio, music, captions.
- Every generation is a take; nothing is overwritten.
- Agent plans show model/objects/time/VRAM; approval is required by default; every
  apply is undoable.
- 500 board cards and 200 timeline clips hit 60fps on the reference machine.
- axe 0 critical; full keyboard path; list view for every canvas.
- No hardcoded localhost; one API client; generated contracts.
- CI runs Vitest, Playwright, axe, lint, typecheck, and bundle budget.

## Related

- [[../systems/studio-shell.md]] — target architecture
- [[../research/frontend-video-platforms-2026.md]] — platform evidence
- [[implementation-plan-2026-11.md]] — backend correctness gates
- [[v2-roadmap.md]] — model/architecture destination
- [[../systems/engineering-audit-2026-11.md]] — defects not to inherit
