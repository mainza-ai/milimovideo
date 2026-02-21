---
name: milimo-frontend-state-manager
description: Strict guidelines for the Milimo Video React 18/TypeScript frontend, specifically regarding the 7-slice Zustand God Store architecture, the SSE listener loops, and the "Magnetic V1" CSS-based timeline rendering logic. Use this when building UI features, modifying state synchronization, or addressing Safari playback drift.
---

# Milimo Frontend State Manager Skill

As the Milimo Frontend State Manager, your primary responsibility is maintaining the integrity, performance, and synchronization of the React 18 UI.

## 1. The Zustand "God Store" (`timelineStore.ts`)

Milimo uses a single `timelineStore.ts` built from 7 distinct slices to prevent context re-render thrashing:
- `projectSlice` (Project metadata, save/load)
- `shotSlice` (Storyboard elements, generation status)
- `trackSlice` (Timeline lanes, clip positioning)
- `playbackSlice` (currentTime, playing status, maximum duration cache)
- `uiSlice` (Tooltips, panel visibility)
- `elementSlice` (Project elements gallery)
- `serverSlice` (SSE handling, active jobs dictionary)

**RULES:**
- **No independent React Contexts for global state**.
- Always use the `<Toggle>` component for switches.
- Components subscribing to frequent updates (like `currentTime` from `PlaybackEngine`) MUST use selective `useStore(useShallow(state => ...))` bounds or access `.getState()` directly via `useRef` to avoid React reconciliation loops.

## 2. Magnetic V1 & Free Placement (Timeline Layout)

The timeline UI is entirely DOM/CSS-based absolute positioning (no HTML5 `<canvas>`).

- Centralized in `computeTimelineLayout()` inside `timelineUtils.ts`.
- **Track 0 (V1)** behaves magically: It ignores user-specified `startFrame`. It calculates layout by iteratively appending the `duration` of Shot `N-1` to Shot `N`. Dragging a clip on V1 simply reorders the array.
- **Track 1 (V2) & Track 2 (A1)** are "free placement": Clips are absolutely positioned based on their explicitly defined `startFrame`, allowing overlapping overlays and audio.

## 3. Engine Loop & Safari Audio Fix (`GlobalAudioManager.ts`)

Safari blocks `<audio>` autoplay and suffers from aggressive media desync when syncing HTML5 video to separate audio tags.

1. **The Playback Loop**: `PlaybackEngine.tsx` uses a headless `requestAnimationFrame` to advance `currentTime` via `Date.now()` deltas.
2. **Audio Decoding**: `GlobalAudioManager.ts` pre-fetches `.mp3` files via HTTP Range requests and uses `decodeAudioData` into an in-memory `AudioBufferSourceNode`.
3. **Mute Control**: Audio nodes remain playing but are muted (`gain.value = 0`) instead of being stopped, circumventing Safari user-gesture policies on re-start.
4. **Drift Tolerance**: The frontend hardcodes a 250ms drift tolerance. If the video `<video>` tag current time deviates from the `PlaybackEngine` time by more than 250ms, it forces a `.fastSeek()`.

## 4. SSE & React Hooks

Backend task updates flow exclusively through Server-Sent Events.

- `SSEProvider.tsx` manages a single robust `EventSource` with exponential backoff.
- The Zustand `serverSlice.ts` exposes `handleServerEvent()` which patches the `shotSlice` implicitly on `"progress"`, `"cancelled"`, and `"complete"` messages.
- **Job Poller**: SSE dropouts happen. On initial page mount, `jobPoller.ts` grabs all shots marked `isGenerating: true` and fires explicit `GET /status/{job_id}` calls to hydrate the UI.

## 5. UI Layout Components

- The layout revolves around `Layout.tsx`, wrapping a top toolbar, `CinematicPlayer.tsx` center stage, `VisualTimeline.tsx` sticky at the bottom, and a right-side inspector (`InspectorPanel.tsx`, `MediaLibrary.tsx`, `StoryboardView.tsx` mutually exclusive visible tabs).
- Drag-and-drop between Library and the Inspector is supported via HTML5 drag APIs.
