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
| `state/jobStore.ts` | Active jobs and metrics; isolated from document/playback |
| `state/agentStore.ts` | Plan history and pending agent proposals |
| TanStack Query (`state/queryClient.ts`) | Health, models, pipelines, review/share, renders, assets, elements, metrics |

The document store is **separate** from the legacy `timelineStore`. Studio routes
(`/studio`) use `useStudioDocumentStore`; legacy routes (`/`) use
`useTimelineStore`. Switching routes does not carry unsaved edits.

**Persistence:** the editor store persists view/layout preferences (active surface,
board camera/areas/links/notes, inspector/timeline sizes, recipes, flags). The
document store persists the last project reference and rehydrates from the server;
transient job/progress state is not persisted.

**Undo:** document history covers user edits to the project document. Board
move/resize/link/area/note history is a separate board stack. SSE/progress events do
not enter undo.

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
- **SSE jobs** update the Studio job store and prompts job polling; document
  completion updates `videoUrl`/`thumbnailUrl` for the selected take.
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

## Related

- [[architecture.md]] — the system around the stores
- [[studio-shell.md]] — Studio surfaces and layout
- [[nle-timeline.md]] — timeline and playback
- [[data-models.md]] — the TypeScript/backend mirror
- [[sse-eventing.md]] — server events and recovery
