---
type: systems
created: 2026-10-09
updated: 2026-11-17
tags: zustand frontend state-management react performance studio
sources:
  - ../web-app/src/studio/state/
  - ../web-app/src/stores/
related:
  - architecture.md
  - studio-shell.md
  - nle-timeline.md
  - data-models.md
  - sse-eventing.md
---

# Frontend State

## Purpose

There are two state systems during the strangler cutover. The **Studio** owns a
document/editor/job/agent split and server state through TanStack Query. The
**legacy editor** keeps the original 7-slice store. They share the backend project
model but not in-memory documents.

## Studio stores

| Store | Owns |
|---|---|
| `state/documentStore.ts` (`milimo-studio-document`) | Project, shots, scenes, captions, save status/revision/conflict, document undo/redo, playback time/playing, track states, toasts |
| `state/editorStore.ts` | Active surface, story view, selection, board camera/positions/areas/links/notes, pins, minimap, timeline zoom/height, inspector state, command dock, recipes, focus mode |
| `state/jobStore.ts` | Active jobs (authoritative replace), finished history, unread count, and metrics; isolated from document/playback |
| `state/agentStore.ts` | Plan history and pending agent proposals |
| `events/jobBridge.ts` | Job→shot reconciliation, stage labels, toasts, desktop notifications, `milimo:job-terminal`/`resync` |
| `notifications/preferences.ts` | Persisted toasts/OS/sound and per-project mute (`milimo-studio-notifications`) |
| TanStack Query (`state/queryClient.ts`) | Health, models, pipelines, review/share, renders, assets, elements, metrics |

The document store is **separate** from the legacy `timelineStore`. Studio routes
(`/studio`) use `useStudioDocumentStore`; legacy routes (`/`) use
`useTimelineStore`. Switching routes does not carry unsaved edits.

**Persistence:** the editor store persists view/layout preferences (active surface,
board camera/areas/links/notes, inspector/timeline sizes, recipes, flags). The
document store persists a **sanitized** project (runtime fields stripped) plus the
saved baseline and save status, debounced by 400 ms and flushed on hide/pagehide;
transient job/progress state is not persisted. The server remains authoritative and
rehydrates on boot.

**Undo:** document history covers user edits to the project document (limit 30 in
Studio). Board move/resize/link/area/note history is a separate board stack.
SSE/progress events do not enter undo, and the undo equality check strips the same
runtime fields as persistence.

**Runtime discipline:** `PlaybackEngine` runs `requestAnimationFrame` only while
playing and caches the clip list/track predicate; a paused seek performs one audio
tick. Polling uses `useVisibleInterval` (one interval per resource, skipped while
hidden, immediate refresh on visibility). `App.tsx` and every surface are lazy, so
the studio shell stays inside the 250 KiB initial-gzip budget.

**Save discipline:** saves are serialized; the top bar shows
Unsaved/Saving/Saved/Save failed/Conflict. A revision mismatch returns 409 and the
UI requires a reload rather than silently overwriting. `documentStore` owns the
CAS/conflict state for Studio.

**Rollback flags:** `VITE_STUDIO_ENABLED=false` serves the legacy editor at
`/studio`. `VITE_STUDIO_DISABLED_SURFACES=shorts,setup` hides individual surfaces;
ModeRail, keyboard switching, and the active-surface guard honor it. The flag
parser is unit-tested.

## Legacy store

`stores/timelineStore.ts` is the original 7-slice Zustand store:
ProjectSlice, ShotSlice, PlaybackSlice, UISlice, TrackSlice, ElementSlice, and
ServerSlice. It uses `milimo-timeline-storage`, persists the project, and tracks the
whole project with zundo. The legacy UI remains at `/` and is frozen except for P0
fixes.

The legacy store still contains the older contract quirks documented in the audit
(camelCase/snake_case mapping, broad project subscribers, and optimistic
`patchShot`). Do not extend it; new work belongs in Studio stores.

## Key Interactions

- **Server reads** flow through TanStack Query or focused hooks; mutations call the
  backend then refresh the affected query.
- **SSE jobs** flow into the Studio job store through `jobBridge`: progress/ETA
  update the shot, a terminal event clears it and refreshes takes/assets/renders,
  and `milimo:job-terminal`/`milimo:resync` trigger reconciliation. Completion
  updates `videoUrl`/`thumbnailUrl` for the selected take.
- **Bootstrap** opens the newest real server project. The local `default` id is a
  sentinel, never a persisted server choice; an explicitly opened UUID is kept.
- **Playback** is isolated in its own clock/store surface so 60fps time changes do
  not reconcile the board or timeline tree.
- **Selection** flows board → shot list → inspector → timeline through editor store
  selection plus document store selected shot.
- **Agent** plans are stored in the agent store; applying a plan goes through the
  document store so it is one undo step.

## Gotchas

- Do not select whole project objects in frequently updating components; select
  narrow fields and use `useShallow` only for related fields.
- The board camera and layout are semantic editor state; order is always
  `sequence_index`, never canvas x/y.
- The Studio and legacy stores can both be mounted in one browser tab during
  navigation; do not assume a shared document.
- The feature flags are build-time Vite env values; changing them requires a
  dev-server restart/rebuild.
- Zod contracts intentionally cover selected endpoints; generated OpenAPI types
  cover the rest. CI checks generated drift.
- `setActiveJobs` from `/active_jobs` replaces the list; `upsertActive` merges an
  immediate local job. A missed terminal must be cleared by the resync/status
  reconcile, not by accumulating active rows.
- Toasts are rendered by the single mounted `StudioToastLayer`; adding another
  container would duplicate notifications. Project mute suppresses toasts/OS/sound
  but history still records every terminal.
- A native EventSource `error` event has no data; `SSEProvider` ignores empty
  payloads and lets `onerror` reconnect.

## Related

- [[architecture.md]] — the system around the stores
- [[studio-shell.md]] — Studio surfaces and layout
- [[nle-timeline.md]] — timeline and playback
- [[data-models.md]] — the TypeScript/backend mirror
- [[sse-eventing.md]] — server events and recovery
