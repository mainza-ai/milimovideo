# Milimo Video — Frontend State (Zustand)

## 1. Store Architecture

The `useTimelineStore` is composed from **7 slices**, wrapped with `persist` (localStorage) and `temporal` (undo/redo via `zundo`).

```mermaid
graph TD
    Store[useTimelineStore]
    
    subgraph "Core Slices"
        Project[ProjectSlice]
        Shot[ShotSlice]
        Playback[PlaybackSlice]
    end

    subgraph "UI Slices"
        UI[UISlice]
        Track[TrackSlice]
    end

    subgraph "Feature Slices"
        Element[ElementSlice]
        Server[ServerSlice]
    end

    subgraph "Middleware"
        Persist["zustand/persist — localStorage"]
        Temporal["zundo — Undo/Redo (20 steps)"]
    end
    
    Store --> Project
    Store --> Shot
    Store --> Playback
    Store --> UI
    Store --> Track
    Store --> Element
    Store --> Server
    Store --> Persist
    Store --> Temporal
```

## 2. Slice Responsibilities

### ProjectSlice
| State | Actions |
|---|---|
| `project: Project` | `setProject(p)` |
| `assetRefreshVersion: number` | `triggerAssetRefresh()` |
| — | `saveProject()` |
| — | `createNewProject(name, settings?)` |
| — | `loadProject(id)` |
| — | `deleteProject(id)` |

### ShotSlice
| Actions | Description |
|---|---|
| `addShot(config?)` | Add new shot with defaults |
| `updateShot(id, updates)` | Local state update |
| `patchShot(id, updates)` | Persist to backend via PATCH |
| `splitShot(id, splitFrame)` | Split at frame position |
| `reorderShots(from, to)` | Reorder in V1 track |
| `deleteShot(id)` | Remove shot |
| `moveShotToValues(id, trackIndex, startFrame)` | Move shot to track/position |
| `addConditioningToShot(shotId, item)` | Add conditioning item (image/video) |
| `updateConditioning(shotId, itemId, updates)` | Update conditioning |
| `removeConditioning(shotId, itemId)` | Remove conditioning |
| `getShotStartTime(shotId)` | Compute absolute start time |
| `generateShot(shotId)` | Dispatch generation request → POST `/generate_advanced` |
| `inpaintShot(shotId, frame, mask, prompt)` | Dispatch inpainting → POST `/edit/inpaint` |

### PlaybackSlice
| State | Actions |
|---|---|
| `currentTime: number` | `setCurrentTime(t)` |
| `isPlaying: boolean` | `setIsPlaying(p)` |

### UISlice
| State | Actions |
|---|---|
| `selectedShotId: string \| null` | `selectShot(id)` |
| `toasts: Toast[]` | `addToast(msg, type)`, `removeToast(id)` |
| `transientDuration: number \| null` | `setTransientDuration(d)` |
| `viewMode: ViewMode` | `setViewMode(mode)` |
| `isEditing: boolean` | `setEditing(e)` |

**ViewMode**: `'timeline' | 'elements' | 'storyboard' | 'images'`

### TrackSlice
| State | Actions |
|---|---|
| `trackStates: Record<number, TrackState>` | `toggleTrackMute(idx)` |
| — | `toggleTrackLock(idx)` |
| — | `toggleTrackHidden(idx)` |

**TrackState**: `{ muted: boolean; locked: boolean; hidden: boolean }`  
**Track mapping**: `0=V1 (Main)`, `1=V2 (Overlay)`, `2=A1 (Audio)`

### ElementSlice
| State | Actions |
|---|---|
| `elements: StoryElement[]` | `fetchElements(projectId)` |
| `generatingElementIds: Record<string, string>` | `createElement(projectId, data)` |
| — | `deleteElement(elementId)` |
| — | `generateVisual(elementId, ...)` → POST `/elements/{id}/visualize` |
| — | `cancelElementGeneration(elementId)` |
| — | `parseScript(text)` → `ParsedScene[]` |
| — | `commitStoryboard(scenes)` |

### ServerSlice
| Actions | Description |
|---|---|
| `handleServerEvent(type, data)` | Processes SSE events and updates relevant shot state |

**SSE Event Types Handled:**
| Event Type | Action |
|---|---|
| `progress` | Update shot `progress`, `statusMessage`, `etaSeconds`, `enhancedPromptResult` |
| `complete` | Set shot `isGenerating=false`, `videoUrl`, `thumbnailUrl`, `progress=100`; trigger `assetRefresh` |
| `error` | Set shot `isGenerating=false`, show toast |
| `cancelled` | Set shot `isGenerating=false`, `status="pending"` |

## 3. Component → Store Mapping

```mermaid
graph LR
    subgraph "Components"
        App[App.tsx]
        Layout[Layout.tsx]
        VT[VisualTimeline]
        TT[TimelineTrack]
        TC[TimelineClip]
        AC[AudioClip]
        CP[CinematicPlayer]
        PBE[PlaybackEngine]
        IP[InspectorPanel]
        PM[ProjectManager]
        ML[MediaLibrary]
        EP[ElementPanel]
        IV[ImagesView]
    end

    subgraph "Selectors Used"
        S_toasts["state.toasts"]
        S_project["state.project"]
        S_playback["state.isPlaying, state.currentTime"]
        S_selected["state.selectedShotId"]
        S_viewMode["state.viewMode"]
        S_elements["state.elements"]
        S_tracks["state.trackStates"]
        S_assetVer["state.assetRefreshVersion"]
    end

    App --> S_toasts
    Layout -->|useShallow| S_project
    Layout --> S_viewMode
    VT -->|useShallow| S_project
    VT --> S_playback
    TT -->|useShallow| S_tracks
    CP -->|useShallow| S_project
    CP --> S_playback
    CP --> S_selected
    PBE --> S_project
    IP -->|useShallow| S_selected
    PM -->|useShallow| S_project
    ML --> S_project
    ML --> S_assetVer
    EP --> S_elements
    IV --> S_assetVer
```

## 4. Middleware Configuration

### Persistence (`zustand/persist`)
```typescript
{
    name: 'milimo-timeline-storage',
    partialize: (state) => ({ project: state.project }),
    merge: (persisted, current) => ({
        ...current,
        ...persisted,
        // Reset transient states on load
        toasts: [],
        isPlaying: false,
        isEditing: false,
        transientDuration: null,
        generatingElementIds: {},
    }),
}
```
Only `project` is persisted to localStorage. All transient state is reset on hydration.

### Undo/Redo (`zundo`)
```typescript
{
    limit: 20,
    partialize: (state) => ({ project: state.project })
}
```
Only project state changes are tracked for undo/redo. Limited to 20 steps.

### Last Project Recovery
```typescript
const getLastProjectId = (): string | null => {
    return localStorage.getItem('milimo_last_project_id');
};
```

## 5. Performance Optimization Strategies

### A. Granular Selectors
```typescript
// ❌ BAD — Re-renders on ANY state change
const { currentTime } = useTimelineStore();

// ✅ GOOD — Re-renders only when currentTime changes
const currentTime = useTimelineStore(state => state.currentTime);
```

### B. `useShallow` for Multi-Selector
```typescript
const { project, viewMode, saveProject } = useTimelineStore(
    useShallow(state => ({
        project: state.project,
        viewMode: state.viewMode,
        saveProject: state.saveProject,
    }))
);
```

### C. Transient Duration
During clip drag operations, `transientDuration` is updated instead of the full `project` structure. This prevents deep re-renders of the entire timeline tree.

### D. Headless Subscriptions
`PlaybackEngine` and `VideoSurface` use `useTimelineStore.subscribe()` for non-rendering updates:
```typescript
// VideoSurface: Drift correction without re-renders
useEffect(() => {
    const unsub = useTimelineStore.subscribe((state) => {
        // Direct DOM manipulation — no React re-render
        videoElement.currentTime = computedLocalTime;
    });
    return unsub;
}, []);
```

### E. GPU-Accelerated Layout
`TimelineClip.tsx` uses `transform: translateX()` for clip positioning (GPU compositing layer) instead of `left` which triggers layout recalculation.

### F. Memo'd Sub-Components
`CinematicPlayer` is decomposed into `memo`'d components to prevent cascading re-renders:
- `VideoSurface` — only re-renders on `shot`/`isPlaying`/`fps` changes
- `PlayerHUD` — only re-renders on resolution/fps/seed changes
- `LoadingOverlay` — only re-renders on generation state changes
- `ControlsBar` — only re-renders on play/edit state changes

## 6. Data Flow: SSE → Store

```mermaid
sequenceDiagram
    participant Backend as EventManager
    participant SSE as SSE Connection
    participant Server as ServerSlice.handleServerEvent
    participant Shot as ShotSlice.updateShot
    participant UI as UI Components

    Backend->>SSE: broadcast("progress", {job_id, progress, status})
    SSE->>Server: handleServerEvent("progress", data)
    Server->>Server: Find shot by lastJobId match
    Server->>Shot: updateShot(shotId, {progress, statusMessage, etaSeconds})
    Shot->>UI: Zustand selector triggers re-render

    Note over Backend: On completion
    Backend->>SSE: broadcast("complete", {job_id, url, thumbnail_url, asset_id})
    SSE->>Server: handleServerEvent("complete", data)
    Server->>Shot: updateShot(shotId, {isGenerating: false, videoUrl, thumbnailUrl})
    Server->>Server: triggerAssetRefresh() — bumps assetRefreshVersion
    Shot->>UI: CinematicPlayer loads new video, MediaLibrary refreshes
```
