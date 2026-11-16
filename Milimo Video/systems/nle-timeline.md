---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: nle timeline zustand playback frontend
sources:
  - docs/01_system_architecture.md
  - docs/user_guide.md
related:
  - architecture.md
  - frontend-state.md
---

# NLE Timeline

## Purpose

The professional editing surface: a magnetic multi-track timeline where AI-generated
shots become an editable film. CSS-based (no canvas) with GPU-composited clip
positioning for 60fps responsiveness.

## Components

| Component | Role |
|---|---|
| `timelineStore.ts` | Single Zustand store (7 slices) — the source of truth; persist + zundo undo/redo |
| `VisualTimeline.tsx` | Renders tracks, clips, playhead; handles drag-and-drop |
| `TimelineTrack.tsx` / `TimelineClip.tsx` | Track and clip rendering; clips use `translateX` for GPU compositing |
| `PlaybackEngine.tsx` | Headless `requestAnimationFrame` loop driving `currentTime`; caches layout in a `useRef` to avoid per-frame GC |
| `GlobalAudioManager.ts` | Web Audio API singleton (`AudioContext` + `AudioBufferSourceNode`); buffers decoded upfront; 0.3s drift tolerance |
| `timelineUtils.ts` | `computeTimelineLayout()` — centralized layout with magnetic V1 logic |
| `snapEngine.ts` | Snap-to-grid/clip/playhead during drags |
| `CinematicPlayer.tsx` | "Program monitor": `VideoSurface`, `PlayerHUD`, `LoadingOverlay`, `ControlsBar` |

## Tracks

- **V1 (magnetic)** — the main story track; clips snap together, no gaps.
- **V2 (overlay)** — free placement; b-roll, inserts.
- **A1 (audio)** — music and SFX, with waveform display (WaveSurfer; Safari gets the
  `MediaElement` backend, others `WebAudio`).

## Key Interactions

- `CinematicPlayer` subscribes to the headless `PlaybackEngine` with throttled drift
  correction (250ms, `fastSeek()` support); video starts muted for Safari autoplay
  policy and un-mutes on first user gesture.
- SSE `complete` events and `jobPoller` keep timeline state in sync with backend jobs.
- Keyboard: `Space` play/pause, `Cmd+Z` / `Cmd+Shift+Z` undo/redo, `Cmd+S` save,
  `Delete` removes selected shot, drag-and-drop imports media.

## Gotchas

- Per-frame work must not recompute layout — the `useRef` cache on
  `computeTimelineLayout` exists because recomputing every frame caused GC pressure.
- Safari needs muted-start + `MediaElement` audio backend; Chrome is fine on WebAudio.

## Target State (2026-11-16)

The replacement timeline is specified in [[studio-shell.md]] and
[[../roadmap/frontend-refactor-2026-11.md]]: V1 magnetic from `order_index`, free
V2/A1/A2/captions, DOM handles + canvas media layer, embedded V1/V2 audio,
keyframes, transitions, and a render graph that matches the cut. Stop adding
features to the current DOM-only timeline except P0 fixes.

## Audit Notes (2026-11-16)

The timeline *renders* correctly in the browser but does not yet *persist* or
*export* correctly. See [[engineering-audit-2026-11.md]] for evidence:

- V1 order is array order only; `PUT /projects/{id}` has no order field and `GET`
  sorts by `created_at`, so reorder is lost on reload.
- Export computes V1 starts from `start_frame`, not magnetic layout; every V1 shot
  in the live DB has `start_frame=0`, so export overlays them all at `t=0`.
- "Push to timeline" sets `trimOut = numFrames`, producing zero-length clips.
- Split shot is broken (missing `datetime` import in the route; client maps a JSON
  string as an array) and assigns the new shot the original index.
- V2 export pads opaque black over V1 (no alpha compositing); V1/V2 embedded audio
  is never mixed. Preview starts muted for Safari autoplay and has no per-clip
  audio control.
- `GlobalAudioManager` ignores `trimOut` when computing clip duration.
- Per-frame `setCurrentTime` still notifies components that select `project`/
  `shots`, so 60fps scrubbing reconciles large trees.

## Related

- [[engineering-audit-2026-11.md]] — timeline/export findings
- [[architecture.md]] — the full system picture
- [[frontend-state.md]] — the store driving the timeline
