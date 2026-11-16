---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: zustand frontend state-management react performance
sources:
  - docs/04_frontend_state.md
  - skills/skills/milimo-frontend-state-manager/SKILL.md
related:
  - architecture.md
  - nle-timeline.md
  - data-models.md
  - sse-eventing.md
---

# Frontend State

## Purpose

One Zustand "God Store" (`useTimelineStore` in `stores/timelineStore.ts`) holds all
global UI state. The standing rule from the project's skill: **no independent React
Contexts for global state** — new state goes into one of the 7 slices, never a new
context.

## Components

| Slice | Owns |
|---|---|
| **ProjectSlice** | `project`, `assetRefreshVersion`; `setProject`, `saveProject()` (PUT upsert), `createNewProject`, `loadProject`, `deleteProject` |
| **ShotSlice** | Shot CRUD + generation ops: `addShot`, `updateShot` (local), `patchShot` (persist), `splitShot`, `reorderShots`, `moveShotToValues`, conditioning items; `generateShot(shotId)` → generation endpoint, `inpaintShot(...)` → `/edit/inpaint` |
| **PlaybackSlice** | `currentTime`, `isPlaying` (+ maximum duration cache) |
| **UISlice** | `selectedShotId`, `toasts`, `transientDuration`, `viewMode` (timeline\|elements\|storyboard\|images), `isEditing` |
| **TrackSlice** | `trackStates` (muted/locked/hidden per track); track map 0=V1 Main, 1=V2 Overlay, 2=A1 Audio |
| **ElementSlice** | `elements`, `generatingElementIds`, `parsedScenes`; `fetchElements`, `createElement`, `generateVisual` → `/elements/{id}/visualize`, `parseScript` (regex) / `aiParseScript` (Gemma 3), `commitStoryboard`, `generateThumbnails`, `pushStoryboardToTimeline` |
| **ServerSlice** | `handleServerEvent(type, data)` — routes SSE events into shot updates |

**Persist** (`zustand/persist`, key `milimo-timeline-storage`): only `project` is
persisted; on hydrate, transient state resets (toasts cleared, `isPlaying=false`,
`isEditing=false`). `milimo_last_project_id` in localStorage drives last-project
recovery.

**Undo/redo** (`zundo` temporal): 20 steps, and only `project` is tracked — UI,
playback, and server state are never undoable.

## Key Interactions

- Server → store flows through `ServerSlice.handleServerEvent`: `progress` updates
  `progress`/`statusMessage`/`etaSeconds`/`enhancedPromptResult` (matched by
  `lastJobId`); `complete` clears `isGenerating`, sets `videoUrl`/`thumbnailUrl`,
  bumps `assetRefreshVersion` (MediaLibrary refreshes); `error` drops a toast;
  `cancelled` resets to pending and clears the job lock. Thumbnail jobs match by
  `shot_id` (not `lastJobId`) and touch only `thumbnailUrl`.
- `patchShot` is **optimistic**: it updates the store first, then PATCHes. It does
  not roll back on a non-2xx response, so a failed save looks successful until
  reload. SSE remains the server → store path for job progress.
- `SSEProvider` owns the single `EventSource`; on mount, `jobPoller` re-syncs any
  shot still flagged `isGenerating` via `GET /status/{job_id}` (see
  [[sse-eventing.md]]).

## Gotchas

- **Selector discipline is load-bearing.** Granular selectors
  (`useTimelineStore(s => s.currentTime)`) plus `useShallow` for multi-field
  selection; `PlaybackEngine` and `VideoSurface` use headless `.subscribe()` so
  60fps updates never trigger React renders. Components subscribing to
  frequent updates must not select broad objects or they create reconciliation loops.
- During clip drags, `transientDuration` updates instead of mutating `project` —
  otherwise the whole timeline tree re-renders per frame.
- **Safari quirk:** `draggable` elements swallow CSS hover events in WebKit —
  interactive buttons inside draggable cards (`ElementCard`, media assets) use
  explicit React `isHovered` state, never CSS `:hover`.
- Magnetic V1: track 0 ignores `startFrame`; layout appends each shot after the
  previous one's duration, so dragging a V1 clip just reorders the array. V2/A1 use
explicit `startFrame` free placement. All in `computeTimelineLayout()`.
- `TrackingPanel` keeps its session state locally, not in the global store.

## Target State (2026-11-16)

This page documents the **current** 7-slice God store. The agreed replacement is
[[studio-shell.md]]: four stores (document / editor / job / agent), TanStack Query
for server state, Immer patches for undo, and one API client. The migration is
[[../roadmap/frontend-refactor-2026-11.md]]. Do not extend the current store with
new global state; put new work in the target shell.

## Audit Notes (2026-11-16)

See [[engineering-audit-2026-11.md]] for evidence:

- **Contract loss:** save sends `frameIndex`, load reads `frame_index`; height,
  per-shot fps, `shotType`, `index`, and `status` are omitted from `saveProject`.
- **74 hardcoded `http://localhost:8000` URLs** across 20 files; `VITE_API_URL` is
  only honored by `ModelSettings`. Remote/Tauri clients break.
- **Autosave races:** a 2s debounced whole-project PUT has no revision or
  serialization; overlapping saves can land out of order.
- **Undo pollution:** zundo tracks the whole project incl. progress/status, so
  server-driven events consume the 20-step history.
- **Stale generation state:** `isGenerating` persists in localStorage; hydrate only
  resets toasts/playback/editing, and `jobPoller` ignores 404.
- **`pushStoryboardToTimeline` lives in `ShotSlice`**, not `ElementSlice`, and sets
  `trimOut = numFrames` (zero-length clips).
- **`patchShot` rollback missing**, as noted above.
- **Drag preview reads undefined `clip.numFrames`/`clip.name`** in `TimelineClip`.

## Related

- [[engineering-audit-2026-11.md]] — frontend findings
- [[architecture.md]] — the system around the store
- [[nle-timeline.md]] — the timeline this state drives
- [[data-models.md]] — the TypeScript types mirror
- [[sse-eventing.md]] — how server events reach the store
