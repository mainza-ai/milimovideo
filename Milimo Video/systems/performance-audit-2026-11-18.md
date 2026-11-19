---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: performance benchmark mps mlx ltx wan ffmpeg react sqlite roadmap
sources:
  - Local benchmarks on M3 Max 128 GB, macOS 27.3, torch 2.10, ffmpeg 9.0.2
  - https://github.com/Lightricks/LTX-2
  - https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/optimization.md
  - https://github.com/Lightricks/LTX-2/issues/304
  - https://github.com/coeyai/ltx-2-mlx-speed-study/blob/main/STUDY.md
  - https://rapidmlx.com/docs/models/families/video
  - https://docs.pytorch.org/docs/2.15/notes/mps.html
  - https://ffmpeg-cookbook.com/en/articles/hardware-encode-videotoolbox
  - https://sqlite.org/wal.html
  - https://react.dev/learn/react-compiler/introduction
  - https://developer.chrome.com/docs/web-platform/best-practices/webcodecs
  - https://fastapi.tiangolo.com/async/
related:
  - ../roadmap/performance-plan-2026-11-18.md
  - memory-management-audit-2026-11.md
  - model-landscape-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
---

# Performance Audit — 2026-11-18

## Verdict

The **UI shell is fast** for its scale; the **generation and render lanes are the
bottleneck**, and several of their costs are structural rather than hardware
limits:

1. The video path runs a **40-step CFG two-stage schedule** while loading a
   **distilled** LTX checkpoint. The official distilled pipeline runs 8+4 steps
   with no CFG. This is the largest single lever.
2. The vendored LTX loader uses the safetensors **mmap** path, which measured
   **~2x slower and 2.5x less stable** than `pread` on MPS upstream. That issue
   was fixed in LTX v1.6.0 with `preadv`.
3. Export still encodes with **libx264 software** while this machine has
   VideoToolbox. A 10s 1080p test: x264 0.86s wall/8.8s CPU; VideoToolbox 0.99s
   wall/1.0s CPU, smaller output, VMAF 97.3 against x264.
4. The web app ships **all eight surfaces eagerly** (334 KiB gzip initial JS) and
   `PlaybackEngine` filters up to 500 shots **every RAF even when paused**.
5. SQLite runs in **DELETE/FULL** journal mode, not WAL, and the bundled SQLite
   3.45.3 predates the 3.51.3 WAL-reset fix.

No app generation benchmark exists: `benchmark_models.py` reports estimates only
(86.8 GB video, 43.0 GB image). That is the first gap to close.

## Machine and stack baseline

| Item | Value |
|---|---|
| CPU / RAM | Apple M3 Max, 16 cores, **128 GB** unified |
| OS | macOS 27.3 |
| Python / torch | 3.12.3 / **2.10.0**, MPS available |
| diffusers | 0.40.0 |
| Node / Vite | 26.5 / 7.x |
| FFmpeg | **9.0.2**, VideoToolbox enabled |

## Local benchmarks

| Area | Measurement | Notes |
|---|---|---|
| Board, 500 cards | **58.5 fps** rAF, **58.0 fps** pan, 11 cards mounted | `perf.spec.ts`; culling works |
| Timeline, 200 clips | **306.1 fps** scroll, 26 clips mounted | far above budget |
| LUT preview 1024×576 | **27.3 fps** (30fps cap), redraws frames | video LUT pixel test |
| Initial JS | **1,684 KiB raw**, 580 KiB largest; **178 + 156 KiB gzip** for index + StudioApp | all surfaces eager |
| API GET project, 500 shots | **17.2 ms** median, 19.8 ms max | local SQLite |
| API PUT project, 500 shots | **71.7 ms** | one transaction |
| API `/health` | 11.3 ms median, **44.2 ms max** | memory sampling |
| API takes, 500-shot project | 5.8 ms median, 93.2 ms max | outlier |
| 10s 1080p encode, x264 fast crf18 | **0.86 s wall / 8.78 s CPU**, 12.3 MB | 300 frames |
| 10s 1080p encode, h264_videotoolbox q65 | **0.99 s wall / 1.02 s CPU**, 10.7 MB | VMAF 97.3 vs x264 |
| Model estimate (video) | **86.8 GB** peak for 49f 512×320 | `benchmark_models.py` estimate, not measured |

The API is not the problem. The board and timeline meet the 60fps product budget.
The render encode is CPU-heavy; VideoToolbox trades a small wall-time change on
this easy clip for ~9x less CPU, which matters when the backend/UI share the
machine.

## What the research says

| Source | Date | Finding that changes this plan |
|---|---|---|
| [Lightricks LTX-2 README](https://github.com/Lightricks/LTX-2) | current 2026 | **DistilledPipeline = 8 predefined sigmas (8 + 4 steps)**, DFR is the production-quality path; single-stage is the fastest prototyping path |
| [LTX-2 optimization docs](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/optimization.md) | current 2026 | `torch.compile` opt-in; gradient estimation allows **20–30 steps instead of 40**; FP8/attention backends are CUDA-targeted; skip inter-stage cleanup when memory allows; MPS DiffVAE falls back to pure-torch SDPA at ~4% cost |
| [LTX-2 issue #304](https://github.com/Lightricks/LTX-2/issues/304) | Aug–Nov 2026 | safetensors **mmap is ~2x slower** than `pread` on MPS and varies 2.5x; **216s → 82s** on a small distilled run; fixed in v1.6.0 with `preadv` |
| [M3 Max MLX speed study](https://github.com/coeyai/ltx-2-mlx-speed-study/blob/main/STUDY.md) | Sep 2026 | 768×512×121 baseline **183s**, stage 2 98s; tiled decode cuts peak 53.4→26.1 GB; cache limit 0 cuts 44.2→25.8 GB; q8 within 13% of bf16, int4 no faster; attention is 10–20% of a block; sparse/windowed attention hurts quality; preview/finish saves 18–20% |
| [Rapid-MLX video docs](https://rapidmlx.com/docs/models/families/video) | 2026 | LTX-2.5 q8 serves on a **24 GB** Mac via the low-RAM distilled path; jobs serialized; **steps are the biggest lever** |
| [PyTorch MPS docs](https://docs.pytorch.org/docs/2.15/notes/mps.html) | Oct 2026 | MPS has no float64; no blanket performance guarantee. Community reports bf16 paths can be slower than fp16 — must A/B on this machine |
| [FFmpeg VideoToolbox guide](https://ffmpeg-cookbook.com/en/articles/hardware-encode-videotoolbox) | Oct 2026 | VideoToolbox uses the media engine, has no CRF, uses `-q:v` 1–100 on Apple Silicon, usually faster than libx264 medium, needs higher bitrate for equal quality |
| [SQLite WAL](https://sqlite.org/wal.html) | Aug 2026 | WAL readers don't block writers; `synchronous=NORMAL` avoids per-commit fsync; **WAL-reset bug fixed in 3.51.3** (current 3.45.3 is older) |
| [React Compiler](https://react.dev/learn/react-compiler/introduction) | current 2026 | Stable; automatic memoization of components/calculations; existing manual memo can remain; incremental adoption |
| [Chrome WebCodecs](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs) | current 2026 | Hardware codecs, `VideoFrame`, worker-friendly, canvas/WebGL interop — the right base for proxies and frame-accurate preview |
| [FastAPI async](https://fastapi.tiangolo.com/async/) | current 2026 | sync routes run in a threadpool; async routes must not block; CPU work belongs in workers |

## P1 findings

### PERF-01 — Distilled checkpoint is driven by a 40-step CFG schedule

`model_engine.py:36` selects `ltx-2-19b-distilled.safetensors`; `video.py:145`
defaults to **40 steps**, and `TI2VidTwoStagesPipeline` stage 1 uses
`guider_denoising_func` (CFG, two transformer passes per step)
(`ti2vid_two_stages.py:130-166`). Official LTX uses `DistilledPipeline` with
**8 sigmas + 4 stage-2 steps and no CFG**.

Approximate transformer evaluations per clip: current stage 1 ≈ 80 passes
(40×CFG) + 4 stage-2 steps; official distilled ≈ 8 + 4. This is the biggest
single speed lever, with a quality decision attached.

### PERF-02 — No measured generation baseline or quality gate

`backend/benchmark_models.py` writes only estimated peak memory; measured
load/wall/peak are blank. Without a real baseline, every optimization is
unverifiable and regressions are invisible. The repo has UI perf specs but no
generation/encode performance gate.

### PERF-03 — MPS weight loading uses the slow mmap path

The vendored loader calls `safetensors.safe_open(... )` with no backend
(`ltx_core/loader/sft_loader.py:30`). Upstream measured **2x** slower warm reads
and **2.6x** end-to-end on a small MPS run, with 2.5x run-to-run variance.
Lightricks fixed it in v1.6.0 with positional `preadv`; this tree still has the
old path.

### PERF-04 — No MLX/q8 lane despite the strongest Apple Silicon evidence

The app only uses PyTorch MPS. The M3 Max MLX study reports **26.0 GB** peak for
768×512×121 after spatial decode tiles and cache bounding, and q8 within 13% of
bf16. The app's own estimate for a smaller video job is **86.8 GB**. An optional
MLX engine (Rapid-MLX/ltx-2-mlx) could unlock longer clips and lower memory even
if it is not the default.

### PERF-05 — VAE decode and residency are not tuned for MPS

- `model_engine.py` converts the MPS VAE to **float32** to avoid black output,
  doubling VAE memory and slowing decode.
- LTX-2 uses default tiling; LTX-2.5 uses `enable_tiling()` but not spatial
  decode tiles/allocator limits.
- The MLX study shows the decode peak (53 GB) can be the largest object; spatial
  tiles cut it to 26 GB for ~2s of extra decode, and a bounded cache cuts
  another ~18 GB.
- Inter-stage cleanup runs unconditionally even with 128 GB.

### PERF-06 — Export uses software libx264 only

`render_utils.py:393` hardcodes `libx264`; model exporters use PyAV libx264.
This machine has `h264_videotoolbox`, `hevc_videotoolbox`, and
`prores_videotoolbox`. Local test: VideoToolbox used **1.02s CPU vs 8.78s**
and produced a smaller file at VMAF 97.3. The wall time on a trivial clip was
similar; the benefit is CPU/power/throughput under load, not a guaranteed wall
win. It needs a quality preset and VMAF gate, not a blind swap.

### PERF-07 — No proxy/transcode lane for playback

Library, Story, Edit, and Compare play the original files. There is no 720p
H.264/HEVC proxy, no WebCodecs decode path, and no pre-warmed frame cache.
4K or long-GOP originals will stutter and burn CPU during review. Chrome's
WebCodecs + worker model is the documented route for frame-accurate editing.

### PERF-08 — Static MPS limits are not machine-aware

`MPS_MAX_RESOLUTION_W/H = 1280/768`, `MPS_MAX_PIXELS ≈ 983K`, and the 0.85
ceiling are hard-coded. On this 128 GB M3 Max they may under-use the machine;
on a 32 GB Mac they may still be too high for some presets. There is no measured
resolution/step/quality ladder per RAM tier.

### PERF-09 — All surfaces ship eagerly

`StudioApp.tsx` imports Story, Generate, Edit, Review, Library, Deliver, Shorts,
Setup, Inspector, Comfy, and LLM settings eagerly; only Previz3D is lazy. Initial
JS is 1,684 KiB raw / ~334 KiB gzip. Lazy surfaces and manual chunks should cut
first-load work without changing behavior.

### PERF-10 — PlaybackEngine does O(shots) work every RAF, even paused

`PlaybackEngine.tsx` runs an unconditional `requestAnimationFrame` loop; each
frame filters `project.shots` for audio twice and recomputes solo/mute state,
then calls `GlobalAudioManager.tick`. At 500 shots that is ~60k filter iterations
and allocations per second while idle in Edit. Cache audio clips/track state and
stop the loop when paused; resume on play.

### PERF-11 — Independent polling is not visibility-aware

Active jobs 2.5s, renders 4s, health 10s, review 15s, memory interval, Comfy
settings; none check `document.hidden`. Several hooks can poll the same data.
Consolidate by resource, pause when hidden, and refresh on visibilitychange/resync.

### PERF-12 — SQLite runs DELETE/FULL, not WAL

`PRAGMA journal_mode` returns `delete`; `synchronous` is `2` (FULL). WAL would
let reads proceed during writes and `NORMAL` removes per-commit fsyncs. Do not
enable WAL blindly: the bundled SQLite is **3.45.3**, older than the 3.51.3
WAL-reset fix. Upgrade SQLite first, add `busy_timeout`, and test one writer.

### PERF-13 — No React Compiler pilot

React 19.2 is installed but the compiler is not. Manual `useMemo`/`memo` is
scattered across BoardCanvas/Timeline. React Compiler is stable and can be tried
incrementally on a surface or route, measured against the perf specs, without
removing existing memoization first.

### PERF-14 — Thumbnail generation can spike `/uploads`

`/uploads` lazily runs ffmpeg to create missing video thumbnails and commits per
asset inside the request loop (`assets.py:333-340`). The first Library load after
importing many videos can block on ffmpeg. Generate posters asynchronously on
upload/job completion and cache them.

### PERF-15 — Health and memory polling do work per request

`/health` measured 11.3ms median / 44.2ms max because it samples memory; it is
polled every 10s plus status surfaces. Cache the memory snapshot for a short TTL
or serve stale-while-refresh; keep a liveness endpoint cheap.

### PERF-16 — LUT preview is a 30fps WebGL→2D copy

The preview hits its 30fps cap at 1024×576 (27.3fps measured). That is fine for
grading, but it uses a full frame upload + `drawImage` copy every frame plus a
second cache entry for the source. If preview is expected to be 60fps, use one
WebGL canvas (or OffscreenCanvas worker) and avoid the 2D copy.

## P2 findings

- **PERF-17 — No HTTP compression middleware.** Large project JSON is served
  uncompressed; local loopback hides it, Tauri/remote LAN does not.
- **PERF-18 — Prompt embeddings and model residency not cached across renders.**
  The MLX study saves 8–9s per repeated prompt with an embedding cache and keeps
  the transformer resident; the app unloads on idle TTL and has no prompt cache.
- **PERF-19 — No hardware decode/encode for render input.** No
  `-hwaccel videotoolbox`/`-hwaccel_output_format` path; export decodes
  software and re-encodes software.
- **PERF-20 — Performance budgets cover only UI and bundle.** No generation
  wall/peak/load budgets, no render encode throughput, no CI benchmark artifact.
- **PERF-21 — Board/minimap and timeline recompute broadly on project changes.**
  Culling is good, but the project array identity changes on every shot edit;
  memoization keyed to shots/fps/track state should be audited if edits stutter
  at 500 cards.
- **PERF-22 — No p95 budgets for API endpoints.** Current medians are fast, but
  the 93ms takes outlier and 44ms health max show tail work (I/O/memory) with no
  budget.

## What is already fast

- Board culling: 11/500 cards mounted; 58 fps pan.
- Timeline culling: 26/200 clips mounted; 306 fps scroll.
- API CRUD at 500 shots: 17ms GET, 72ms PUT.
- LUT preview: correct pixels at ~30fps; no taint/black stage.
- Bundle under budget and gzip ~334 KiB initial.
- GPU queue/memory admission prevents concurrent model OOM.

## What not to do (from the research)

- **Don't chase sparse/windowed attention first.** The MLX study measured
  attention at 10–20% of a stage-2 block; a perfect sparse kernel bounds at
  9–16%, and their windowed attempt produced a ghosted image.
- **Don't assume int4 is faster than q8.** It measured no faster; q8 is within
  13% of bf16 with much less memory.
- **Don't enable `torch.compile` on MPS blindly.** MPS compile has known
  regressions (4–9x slower conv cases in PyTorch issues); pilot and measure.
- **Don't swap bf16→fp16 without quality checks.** MPS bf16 paths are reported
  slower than fp16 in some workloads, but the LTX pipeline is bf16-native;
  A/B wall time, peak, and SSIM/VMAF.
- **Don't use VideoToolbox without a VMAF gate.** It has no CRF and needs a
  quality preset/bitrate decision.

## Recommended direction

1. **Measure first.** A repeatable, machine-tagged benchmark of load, stage 1,
  stage 2, VAE, encode, and peak for fixed prompts/resolutions. Store results
  as an artifact and gate regressions.
2. **Use the distilled schedule.** Add a Draft preset (8+4 distilled, no CFG)
  and a Final path (CFG/two-stage) with an explicit quality comparison.
3. **Fix weight loading.** Adopt the upstream `preadv` loader (or upgrade the
  vendored LTX to v1.6.0); this is a low-risk, large fixed-cost win.
4. **Treat memory as the enabler.** Spatial VAE tiles + allocator/cache limits,
  keep the transformer resident, cache prompt embeddings, and test whether the
  float32 VAE workaround is still needed on torch 2.10.
5. **Add an MLX lane as an experiment, not a rewrite.** Target LTX-2.5 q8 on
  this M3 Max; compare peak/wall/quality against PyTorch. The external study
  already reports 26GB vs the app's 86.8GB estimate.
6. **Hardware encode with a quality gate.** Add VideoToolbox H.264/HEVC
  presets, VMAF/SSIM checks, and choose software for archival.
7. **Proxies and WebCodecs for review.** Generate 720p proxies on ingest and
  use WebCodecs/WebGPU where available; keep the original for export.
8. **Trim frontend idle work.** Lazy surfaces, cached PlaybackEngine inputs,
  paused RAF, visibility-aware polling, React Compiler pilot.
9. **Make SQLite boringly fast.** Upgrade SQLite, enable WAL + `synchronous=
  NORMAL`, set `busy_timeout`, one writer, cached health.
10. **Add performance gates to CI.**

## Implementation plan

Phased performance plan with acceptance criteria, effort, and dependencies:
[[../roadmap/performance-plan-2026-11-18.md]].
