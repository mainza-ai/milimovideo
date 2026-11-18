---
type: systems
created: 2026-10-09
updated: 2026-11-18
tags: nle timeline studio playback render ffmpeg
sources:
  - ../docs/user_guide.md
  - ../web-app/src/studio/
  - ../backend/utils/render_utils.py
related:
  - architecture.md
  - studio-shell.md
  - frontend-state.md
---

# NLE Timeline

## Purpose

The editing surface where AI-generated shots become a film. The Studio timeline is
a DOM-handle timeline with an isolated playback clock, a keyboard/table equivalent,
and a render graph that matches the cut. The legacy editor keeps its original
DOM timeline as the frozen fallback.

## Studio components

| Component | Role |
|---|---|
| `studio/state/documentStore.ts` | Studio project document, save/revision, document undo |
| `studio/components/StudioTimeline.tsx` | Tracks, clips, tools, trim gestures, audio picker, caption lane, table view |
| `studio/components/PlaybackEngine.tsx` | Studio-owned `requestAnimationFrame` clock, separate from the legacy store |
| `studio/components/Waveform.tsx` | Audio waveform rendering |
| `studio/components/MotionPanel.tsx` | Keyframe editing for opacity/scale/pan/volume |
| `studio/utils/timelineUtils.ts` / `timeline/ops.ts` | Shared computed layout and pure edit operations |
| `backend/utils/render_utils.py` | FFmpeg filtergraph builder for the computed cut |

## Tracks

- **V1** — magnetic story track. Order is `sequence_index`; `start_frame` is ignored.
- V1/V2 clips show the shot poster behind the title; uploaded and generated videos
  get poster frames.
- **V2** — free overlay track with alpha compositing.
- **A1 / A2** — free audio/music tracks, each with mute/solo and OTIO export.
- **C1** — caption lane backed by project caption cues.
- **Titles** — project-level burned-in text overlays, not a track.

## Tools and interactions

- **Select/trim** — click clips; drag edge handles to trim. Handles support keyboard
  nudging through the table path.
- **Slip** — move the source window while keeping duration.
- **Roll** — move a cut together with the next V1 clip.
- **Blade** — split at the playhead (`B`).
- **Ripple delete** — remove and reflow V1 (`R`).
- **Slide** — move the selected V1 clip earlier/later (`[` / `]`).
- **Rate** — 0.25–4×; duration changes and render applies `setpts`/`atempo`.
- **Audio** — gain, mute, fade-in/out, per-clip solo/mute, A1 placement, waveform.
- **Captions** — add at playhead, edit in/out/text, import SRT/VTT, optional burn-in.
- **Motion** — opacity, scale, pan X/Y, and volume keyframes.
- **Table** — editable rows for clips, trims, speed, mute, and delete; keyboard edit
  path for accessibility.
- **Range** — deliver in/out seconds; the render respects it.

## Render truth

The export is computed from `sequence_index` and placement, not from canvas
coordinates or stale `start_frame` values:

- magnetic V1 timing across the project;
- V2 alpha-correct overlay over V1;
- embedded V1/V2 audio plus A1 (`amix duration=longest`);
- per-clip gain/mute, fades, solo, and track mute/hidden state;
- speed, trim, opacity/scale/pan/volume keyframes;
- caption ASS burn-in;
- render range and fit (`pad` or `crop`);
- persisted project defaults for CRF, encoder speed, audio bitrate, and optional
  single-pass `loudnorm` at −16 LUFS.

The render runs as an FFmpeg subprocess with parsed progress, cancellation, a
single terminal event, and a `Job` row. Tests assert real output duration,
dimensions, pixels, audio presence/bitrate, and filtergraph contents.

## Playback

- `PlaybackEngine` owns a studio clock; components subscribe narrowly so scrubbing
  does not reconcile the whole project tree.
- Play/pause is global (`Space` opens Edit when needed).
- Video preview starts muted for Safari autoplay, then unmutes on the first gesture.
- WaveSurfer supplies waveforms; Safari uses the `MediaElement` backend.

## Performance

- 500-card board pan measured ~58 fps headless.
- 200-clip timeline scroll measured ~275–280 fps headless.
- Board culling and timeline culling are committed in Playwright perf specs.
- Avoid per-frame project subscriptions; use narrow selectors and the isolated
  playback clock.

## Legacy timeline

The legacy `/` timeline remains a 3-track DOM editor with its own store and export
modal. It is frozen except for P0 fixes. Do not add new features to it; Studio is the
current shell.

## Titles and A2

Burned-in titles are a project JSON list rendered with `drawtext` after the LUT
and before captions; Deliver exposes text, in/out, size, and color. A2 is a real
second audio lane (`track_index == 3`) mixed with A1 and embedded audio through the
same `amix` graph, with its own mute/solo state and OTIO export/import.

## Crossfades

A per-shot `transition_in` (frames) overlaps a V1 clip with its predecessor and
ramps the incoming alpha from 0 to 1 with `geq`; the outgoing clip stays visible
underneath. The clamp is half of both clips. The timeline layout, render, and OTIO
metadata share the same overlap math. Other transition types (dip, wipe, slide)
are not implemented.

## Grading

A project can carry one 3D LUT (`project.render_lut`), uploaded into
`projects/<id>/assets/lut/` as `.cube`, `.3dl`, `.dat`, or `.m3d`. Deliver resolves
the URL, validates the file, and passes it to the render graph; FFmpeg applies
`lut3d=...:interp=trilinear` to the V1/V2 composite before caption burn-in. The LUT
is persisted with the project and included in the render job settings. A real
pixel test renders a red clip through a red/blue swap LUT and asserts the blue
channel dominates.

## Known scoped gaps

- Only a V1 crossfade; dip/wipe/slide transitions are not implemented.
- No canvas media layer; clips are DOM elements with culling.
- Proxies and media relink are not implemented.

## Related

- [[architecture.md]] — the full system picture
- [[studio-shell.md]] — Studio surface architecture
- [[frontend-state.md]] — stores and playback isolation
- [[../roadmap/implementation-plan-2026-11.md]] — render/scheduler status
