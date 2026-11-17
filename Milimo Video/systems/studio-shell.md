---
type: systems
created: 2026-11-16
updated: 2026-11-17
tags: frontend architecture canvas timeline studio shell ux
sources:
  - wiki/research/frontend-video-platforms-2026.md
  - wiki/systems/engineering-audit-2026-11.md
  - web-app/src/
related:
  - frontend-state.md
  - nle-timeline.md
  - data-models.md
  - sse-eventing.md
  - ../roadmap/frontend-refactor-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
---

# Studio Shell — Implemented Front-End Architecture

## Purpose

Define the Studio shell: a local-first film studio where **the board, the shot, and
the cut are the same project**. It replaces the legacy tab-switched, God-store UI
with one canonical project graph and synchronized lenses:

- **Story** — script, board, shot list.
- **Generate** — shot workspace, ingredients, takes.
- **Edit** — stage, timeline, inspector.
- **Review** — versions, compare, comments, approvals.
- **Library / Deliver** — assets, master references, takes, exports.
- **Shorts** — transcript candidates, reframe, variants.
- **Setup** — device, queue, models, integrity, error budget.

The reference implementation for the canvas interaction vocabulary is
[Kubona](https://github.com/mainza-ai/kubona) (Apache 2.0, inspected at
`6d1e2cb` on 2026-11-16), especially
`SpatialCanvas.tsx`, `Workbench.tsx`, `workbench-model.ts`,
`SpatialCanvas3D.tsx`, `ModeRail.tsx`, `CommandDock.tsx`, `Inspector.tsx`,
`CopilotDrawer.tsx`, and `CommandPalette.tsx`. Borrow the interaction patterns and
small utilities with attribution; do **not** import the quantum domain, cards, or
product language.

## Implementation Status (2026-11-17)

The scoped F0–F5 shell is implemented: Story, Generate, Edit, Review, Library,
Deliver, Shorts, and Setup; board and timeline as peers; immutable takes; agent
plan/approval/undo; review/share; queue pause/resume; seed lock; encoding controls;
integrity scan/repair; Tauri native open/save.

Known scoped gaps: no transitions/LUT/titles/A2, lexical rather than embedding
search, no shorts/lens template packs, and no bundled Python sidecar runtime. Legacy
`/` remains until the two-clean-release cutover.

## Design Stance

1. **One project graph, multiple lenses.** Board position, shot order, timeline
   placement, takes, and review state are separate fields on one canonical model.
   No surface owns a private copy.
2. **Canvas is for meaning; timeline is for time.** The board answers “what is the
   story and how does it relate?” The timeline answers “when and how long?” Never
   use canvas coordinates as sequence order.
3. **The shot card is the atom.** A card, a shot-list row, and a timeline clip are
   three presentations of the same `Shot`. Selecting one selects all.
4. **Takes are immutable.** Every generation is a take with lineage. The shot
   points at the selected take; the review surface compares takes.
5. **The agent proposes, the filmmaker disposes.** Plans show models, affected
   objects, estimated time/VRAM/cost; generation is ask-first by default. Every
   applied agent action is an undoable patch.
6. **Local-first is visible.** No cloud dependency in the hot path; model slots,
   queue, disk, and device are first-class status. The UI should feel like a
   studio appliance, not a SaaS dashboard.

## Surfaces

| Surface | Question it answers | Primary layout |
|---|---|---|
| **Story** | What is the film? | Script / Board / Shot list |
| **Generate** | How do I make this shot? | Shot list + Stage + Takes + Ingredients |
| **Edit** | When does it happen? | Stage + Timeline + Inspector |
| **Review** | Is this the right take? | Versions + Compare + Comments |
| **Library** | What do I own? | Assets / Elements / Takes / Models |
| **Deliver** | How does it go out? | Presets + Variants + Render queue |
| **Shorts** | How do I repurpose the cut? | Transcript + candidates + reframe |
| **Setup** | How is the studio configured? | Device / Models / Queue / Integrity |

The mode rail is intentionally small (six icons + settings). Every surface is
reachable from the command palette and from contextual links (“Open in Edit”,
“Send to Review”, “Find in Library”).

### Workspace presets

- **Film** — Story → Generate → Edit → Review → Deliver.
- **Shorts** — Source/Transcript → Find moments → Reframe/Caption → Variants →
  Deliver.

The preset controls defaults, not permissions. A Film project can export shorts;
a Shorts project can open the timeline.

## Layout System

### Studio shell (default)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Top bar: logo · Project ▾ · Sequence ▾ · Story|Generate|Edit|...    │
│          Save state · Queue · Local/Remote · Share · Needs you (3)  │
├──────┬──────────────────────────────────────────────────┬───────────┤
│ Rail │ Center surface                                   │ Inspector │
│ 56px │ Board / Stage / Timeline / Review / Library      │ 320–480px │
│      │                                                  │ resizable │
├──────┴──────────────────────────────────────────────────┴───────────┤
│ Command dock: context chips · prompt/agent · model · estimate · Run  │
├─────────────────────────────────────────────────────────────────────┤
│ Status strip: device · model slot · queue · disk · autosave · FPS   │
└─────────────────────────────────────────────────────────────────────┘
```

### Story layout

```
┌ Spine (240px) ┬ Board canvas ────────────────────────┬ Inspector ┐
│ Acts          │ [Master References] [Look Dev]       │ Shot      │
│  Scenes       │ [Scene 1: cards + links]             │ Prompt    │
│   Shots       │ [Scene 2: cards + links]             │ Takes     │
│ Filters       │                                      │ Review    │
├───────────────┴──────────────────────────────────────┴───────────┤
│ Board toolbar: zoom · fit · minimap · LOD · edges · arrange · list │
└────────────────────────────────────────────────────────────────────┘
```

Tabs inside Story: **Script**, **Board**, **Shot list**. They share selection and
scroll position where sensible. The board is not a replacement for the shot list;
the shot list is the accessible/bulk-edit view.

### Edit layout

```
┌ Program monitor / compare viewer ───────────────────┬ Inspector ┐
│ overlays: safe area · grid · mask · track · depth   │ Clip      │
│                                                      │ Take      │
├──────────────────────────────────────────────────────┤ Audio     │
│ Timeline (45–60% height)                             │ Effects   │
│ ruler · V2 · V1 · A1 · captions · markers             │ History   │
└──────────────────────────────────────────────────────┴───────────┘
```

### Review layout

```
┌ Versions (220px) ┬ Compare viewer ──────────────────┬ Comments ┐
│ v3 (current)     │ side-by-side / wipe / stack      │ @timecode│
│ v2               │                                  │ threads  │
│ v1               │                                  │ approve  │
└──────────────────┴──────────────────────────────────┴──────────┘
```

### Focus modes

- **Focus canvas** — hide rail, inspector, dock; `Esc` restores.
- **Focus stage** — hide timeline; useful for review/playback.
- **Presentation** — full-screen board or cut with minimal chrome; later.

### Responsive rules

- **≥1440px** — three columns, timeline 45%.
- **1024–1439px** — inspector overlays or collapses; timeline 45–55%.
- **768–1023px** — rail + single surface; inspector as sheet; dock collapses to
  one line.
- **<768px** — **Review mobile**: stage, versions, comments, approvals, job
  status. Full editing is desktop-first; do not pretend otherwise.

Layout state is persisted per project/user: rail width, inspector width/tab,
timeline height, dock height, board camera, timeline zoom/scroll, active surface,
compare mode. Follow Kubona's split: **semantic layout** (card positions/areas)
saved with revision/CAS; **ephemeral view** (camera/selection/zoom) saved
separately. This prevents autosave races and keeps undo clean.

## Canonical Objects

```
Project
 ├── Sequence (film cut; may be multiple: trailer, short, vertical)
 │    ├── Scene
 │    │    └── Shot ── selected_take_id ──▶ Take
 │    │         ├── order_index (story order)
 │    │         ├── timeline_placement {track, start_frame, trim_in, trim_out}
 │    │         ├── prompt / negative / params
 │    │         ├── element_refs[]
 │    │         ├── reference_refs[]
 │    │         └── review_state
 │    ├── Track (V1, V2, A1, A2, captions)
 │    └── Marker / Comment
 ├── BoardLayout
 │    ├── BoardNode {ref, x, y, w, h, area_id, pinned, collapsed}
 │    ├── BoardArea {act/scene/look/master-refs/notes, rect, color}
 │    └── BoardLink {source, target, kind}
 ├── Element / MasterReference (character, location, object, style, voice)
 ├── Asset (uploaded/generated media, audio, model, document)
 ├── Take (immutable generation result)
 ├── Render (deliverable)
 └── Review (threads, approvals, activity)
```

**Rules**

- `order_index` is explicit and persisted. Board x/y never defines sequence.
- `timeline_placement` is explicit; V1 is magnetic (computed from order), free
  tracks use `start_frame`.
- `Take` owns model, seed, params, duration, audio, thumbnail, cost/time, lineage,
  and review state. `Shot.selected_take_id` points at the current take.
- `BoardNode` is presentation metadata only; deleting it never deletes the shot.
- `Element` is promoted to `MasterReference` semantics: images, description,
  trigger/name, voice, strength, usage count.
- JSON columns are replaced by typed child tables or validated JSON with a
  schema version. No `List[dict]` at the API boundary.

## Canvas Engine

### Core model

`SpatialCanvas` is a generic surface with a camera `{x, y, zoom}` and a world layer
transformed with `translate3d + scale`. It provides:

- Pan: drag empty space, space+drag, middle mouse.
- Zoom: wheel/pinch around pointer; fit selection; zoom to 100%.
- Select: click, shift-click, marquee; multi-select toolbar.
- Move/resize: pointer capture, grid snap, alignment guides.
- Arrange: align left/right/top/bottom, distribute, auto-arrange.
- Areas: create/rename/color/collapse/assign; drag children with area.
- Links: typed links; drag from handle; edge visibility modes.
- Pins: pin a card so auto-layout never moves it.
- Minimap: canvas-rendered, click/drag to pan.
- LOD: `summary` (<0.55), `compact` (<0.95), `detail` (≥0.95).
- Culling: render only nodes intersecting viewport + margin.
- Undo: board-local stack for move/resize/link/area/arrange/remove.
- List view: an accessible table equivalent, never a fallback afterthought.
- Persistence: semantic layout save with revision; view save separate.

These are directly modeled on Kubona's Workbench and tldraw's production
requirements (viewport culling, batching, camera, minimap, focus mode, custom
shapes).

### Board lenses (progressive)

1. **Story Board** (implemented) — cards, areas, links, master references, takes,
   notes.
2. **Pipeline lens** (implemented v1) — validation graph for a shot recipe, with
   model/time/VRAM estimate, apply-to-shot, and JSON recipe import/export. It is a
   lens, not the default surface.
3. **Blocking lens** (implemented v1) — top-down camera/character/object diagram
   persisted per shot; reset and rotation controls.
4. **Previz 3D** (implemented v1) — lazy three.js stage with camera path, orbit/zoom/
   pan, and PNG snapshot. WebGL failure is caught; the editor continues.

### Board node morphologies

| Kind | Default size | Detail |
|---|---|---|
| `scene` | 360×220 | slugline, shot count, status, thumbnail strip |
| `shot` | 320×260 | media, prompt, elements, duration, status, takes |
| `take` | 280×200 | thumbnail/video, model, seed, cost, favorite |
| `element` | 240×260 | character sheet, trigger, usage, voice |
| `reference` | 220×180 | image/video, source, tags |
| `note` | 260×160 | rich text, color |
| `task` | 240×140 | owner, status, due |
| `job` | 260×140 | type, progress, ETA, cancel |
| `render` | 300×180 | preset, duration, status, open |

Cards support **summary/compact/detail** content, hover actions, context menu,
drag handle, progress overlay, and selection state. Media is lazy: poster first,
video metadata on visibility, playback only for the active/hovered card.

### Master References area

Pinned top-left area labeled **Master References**. Contains characters,
locations, objects, styles, voices. Each card shows the canonical image/voice,
trigger, description, and usage count. The agent and prompt composer can reference
these by `@Name` or by dragging. This is the Milimo answer to Flow Ingredients,
Kling Elements, and Luma Master Reference Assets.

## Timeline Engine

- **Tracks:** V2 (overlay), V1 (magnetic main), A1/A2 (audio), C1 (captions),
  optional markers. Add tracks later; don't hardcode three in the store.
- **Clip model:** `{shot_id, take_id, track_id, start_frame, duration_frames,
  trim_in, trim_out, gain, fades, keyframes, transition_in/out}`.
- **Rendering:** DOM for headers/handles/labels; canvas layer for dense
  thumbnails/waveforms when clip count exceeds ~200. Virtualize horizontally.
- **Tools:** select, blade, ripple, roll, slip, slide, trim, rate, transition,
  text, mask, track select.
- **Audio:** waveform, gain, fades, mute/solo, ducking; V1/V2 embedded audio is
  first-class; A1/A2 are separate.
- **Captions:** text track tied to transcript; SRT/VTT import/export; burn-in
  option.
- **Keyframes:** opacity/scale/position/volume; graph editor later.
- **Snapping:** clips, playhead, markers, grid; configurable.
- **Undo:** command-based, coalesced for drags; timeline and document histories
  must not fight.
- **Playback:** master clock in a narrow store; `requestAnimationFrame`; drift
  correction; no broad `project` selectors in per-frame subscribers.
- **Render:** range, presets, background queue, granular `ffmpeg -progress`.

The timeline must be usable without the board. The board must be usable without
the timeline. Neither is a modal.

## Stage & Review

### Stage

- **Program** — the cut at the playhead.
- **Shot** — the selected shot/take, looped.
- **Compare** — A/B wipe, side-by-side, stack, onion-skin.
- **Overlays** — safe area, thirds, mask, tracking, depth, camera path, frame
  info.
- **Edit frame** — pause, paint mask, prompt, preview image, then generate video
  (Runway Aleph / Milimo inpaint pattern).
- **Take strip** — thumbnails of all takes for the current shot; click to select.

### Review

- **Version stacks** — group takes by shot/asset; newest first; compare.
- **Comments** — anchored to take + frame/timecode; threads, mentions, resolve.
- **Approvals** — pending/approved/changes requested; per reviewer.
- **Activity** — who changed what, when.
- **Share** — local link with read/comment permissions; no cloud required.
- **Agent in comments** — `@Agent` can propose an edit, but the proposal appears in
  the approval queue, never silently applied.

## Command Dock & Agent

The dock is always available at the bottom of the shell; it has four modes:

1. **Ask** — question/answer over the project; no spend.
2. **Direct** — natural-language edit; produces a plan/diff.
3. **Generate** — prompt + ingredients + model/params + estimate; queues on
   approval.
4. **Find** — semantic search across assets, shots, takes, comments.

### Context chips

The composer always shows what the agent sees: selected shot(s), scene/sequence,
board area, time range, attached references, master references, previous take.
Chips are removable. This solves the “chat disconnected from the document”
anti-pattern.

### Plan-before-spend

A plan card shows:

- Steps and affected objects (shots, clips, elements).
- Model(s), prompt(s), duration/resolution.
- Estimated GPU time/VRAM (local) or credits (cloud).
- A contact-sheet/timeline diff preview.
- Buttons: **Approve**, **Edit plan**, **Dry run**, **Cancel**.

Defaults: **ask before generating**. “Auto” is opt-in per project/session. Local
GPU time is a cost; the estimate is not optional.

### Skills / recipes

A successful plan can be saved as a named **Skill** with typed inputs (e.g.,
“Product ad: 6 shots, 9:16, captions, brand kit”). Skills appear in the command
palette and can be run on a selection. This is the Milimo version of Runway
Skills, Krea Node Apps, and ComfyUI workflows — local JSON, versioned, no cloud.

### Agent scope

V1: read selection/board/timeline, propose prompts, create shots/cards, arrange
timeline, write script beats. V2: run generation plans, compare takes, apply
review fixes. V3: multi-agent director/cinematographer/continuity/reviewer crew
from the v2 roadmap, behind explicit approval.

## State & Data Flow

Replace the 7-slice God store with four stores and server-state hooks:

| Store | Owns | Persistence |
|---|---|---|
| `documentStore` | Project graph: sequences, scenes, shots, elements, takes, timeline placements | Server + local cache; undoable |
| `editorStore` | Active surface, selection, panel sizes, board camera, timeline zoom, compare mode, focus | Local + per-project view endpoint; not undoable |
| `jobStore` | Server-derived jobs/progress; no local truth | TanStack Query + SSE |
| `agentStore` | Chat messages, plan, approvals, streaming trace | Local session + optional transcript file |

Rules:

- Server state is **not** copied into Zustand. Use TanStack Query for
  projects/assets/jobs/models; use SSE to invalidate/patch query data.
- Document edits use **Immer patches** or command objects; undo is patch-based,
  not whole-project snapshots.
- Board layout has its own undo stack and revision; it never pollutes document
  undo.
- Playback state (`currentTime`, `isPlaying`) is a tiny external store or
  `useSyncExternalStore` source; per-frame subscribers must not select `project`.
- Selection is a set of IDs, not object copies. Surfaces resolve IDs from the
  document store.
- Transient generation status lives in `jobStore`, keyed by job/shot; the
  document only stores durable state (`selected_take_id`, `review_state`).

### Save discipline

- Document saves are serialized (single in-flight PUT), revision-checked, and
  coalesced.
- Board semantic layout saves use revision/CAS; view saves are separate and
  debounced.
- Show **Saving / Saved / Conflict / Offline** in the top bar and status strip.
- On conflict, offer “reload latest” or “keep mine” with a diff summary; never
  silently overwrite.

## Contracts & API

- Generate TypeScript from OpenAPI (or a shared schema) and validate with Zod at
  the boundary.
- One API client with base URL from `VITE_API_URL` or `window.location`; no
  hardcoded localhost. SSE client uses the same base and token.
- Pydantic models are the source of truth for `Shot`, `Take`, `TimelineItem`,
  `BoardNode`, `BoardArea`, `BoardLink`, `Comment`, `Review`.
- Add endpoints:
  - `GET/PUT /projects/{id}/board` (semantic layout, revision)
  - `GET/PUT /projects/{id}/view` (camera, selection, zoom, surface)
  - `GET/POST /shots/{id}/takes`, `POST /shots/{id}/select-take`
  - `GET/POST /shots/{id}/comments`, `POST /comments/{id}/resolve`
  - `GET /projects/{id}/activity`
  - `POST /agent/plan`, `POST /agent/apply` (approval + patch)
- All write responses return the new revision; the client rejects stale writes.

## Performance Budgets

| Scenario | Budget |
|---|---|
| Board pan/zoom, 500 cards | 60fps on M1 Pro / RTX 3060 class |
| Timeline scrub, 200 clips | 60fps; no full-tree re-render |
| Selection → inspector | <100ms |
| Board layout save | <150ms p95 local |
| First meaningful paint | <2s local |
| Video preview start | <500ms for cached clip |
| Memory, 500 cards | <300MB renderer |

Techniques: viewport culling, LOD, `content-visibility`, `contain`, transform-only
camera, memoized cards, `useSyncExternalStore`, React Compiler, route-level code
splitting (3D/models/review), lazy video metadata, canvas timeline layer,
virtualized lists, no per-frame layout recompute.

## Accessibility

- **Every canvas has a list/table equivalent** (Kubona workbench list, Boords shot
  list).
- Full keyboard model: `Cmd+K` palette, `Space` play, `J/K/L`, `I/O`, `B` board,
  `E` edit, `G` generate, `R` review, `?` shortcuts, `Esc` focus.
- Focus management for panels/modals; visible focus rings; skip link.
- ARIA live for job progress/completion; no color-only status.
- Reduced motion; disable edge animations and camera glides.
- Contrast tokens; test both dark and high-contrast.
- Captions/transcripts for all generated audio; export SRT/VTT.
- Screen-reader descriptions for cards and timeline clips.

## Design Language

Preserve the Milimo cyan identity, but make it an accent, not a wash:

- **Canvas:** near-black graphite (`#08090a`), panels `#0f1113`, raised `#16191c`.
- **Text:** primary `#e6edf2`, secondary `#9aa7b2`, muted `#6f7c87`.
- **Accent:** cyan `#06b2eb`; use for primary action/selection, not every border.
- **Semantic:** success `#3ddc97`, warning `#f5b84b`, danger `#ff5d6c`,
  info `#6aa9ff`.
- **Tracks:** V1 cyan, V2 violet, A1 green, A2 teal, captions amber.
- **Type:** Geist/Inter for UI, JetBrains Mono for timecode/technical values.
- **Radius:** 6/10/14/20; **shadows:** one float shadow, not glass everywhere.
- **Motion:** 70ms press, 140ms fast, 220ms standard, 340ms slow; honor reduced
  motion.
- **Density:** comfortable by default, compact toggle for edit mode.
- **No glassmorphism as hierarchy.** Use surfaces, borders, and type scale.

## What We Keep vs Replace

**Keep**

- Domain concepts: Project/Scene/Shot/Element/Asset/Job, @triggers, storyboard
  parsing, Quantum Alignment, SAM 3, model library.
- Backend contracts after the audit fixes (canonical order, typed timeline).
- React 19 + Vite + TypeScript + Tailwind 4; native video/audio; wavesurfer.
- The lesson from the current app that a timeline must feel frame-accurate.

**Replace**

- The 7-slice God store and whole-project persist/zundo.
- Hardcoded localhost and duplicated fetch logic.
- Tab-switched center that loses context.
- Modal-driven generation with no takes/plan.
- The current `Layout`/`VisualTimeline`/`Inspector` composition.
- Ad-hoc inline styles and glass utilities as the primary design system.

## Related

- [[../roadmap/frontend-refactor-2026-11.md]] — migration phases
- [[../research/frontend-video-platforms-2026.md]] — external evidence
- [[frontend-state.md]] — current store (to be replaced)
- [[nle-timeline.md]] — current timeline (to be replaced)
- [[data-models.md]] — current persistence (to be extended)
- [[../roadmap/implementation-plan-2026-11.md]] — correctness gates
