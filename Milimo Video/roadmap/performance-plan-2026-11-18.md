---
type: roadmap
created: 2026-11-18
updated: 2026-11-18
tags: performance plan benchmark mps mlx ltx render frontend sqlite
sources:
  - ../systems/performance-audit-2026-11-18.md
  - ../systems/memory-management-audit-2026-11.md
  - ../research/model-landscape-2026-11.md
related:
  - ../systems/performance-audit-2026-11-18.md
  - implementation-plan-2026-11.md
  - open-work-resolution-2026-11.md
---

# Performance Implementation Plan (2026-11-18)

Input: [[../systems/performance-audit-2026-11-18.md]] — 22 findings
(`PERF-01…PERF-22`) across generation, memory, render, frontend, and storage.

**Memory companion:** [[memory-management-plan-2026-11-18]] (M0–M5) owns the
memory truth/allocator/runtime-guard/browser/agentic depth. P2 is this plan's
execution slice for VAE/MLX/machine tiers; M0/M1 feed its measurements, and M4 adds
agentic memory.

Priority: **measure → generation schedule/loader → memory/MLX → render/proxy →
frontend idle work → storage/API → gates**.

Status: **P0/P1 core, P3 encoder/proxy core, P4, P5, and P6 gates
implemented 2026-11-18; model-loaded benchmarks and the VMAF gate remain open.**

## Implementation status (2026-11-18)

| Phase | Landed | Still open / deliberately deferred |
|---|---|---|
| P0 | `backend/benchmark_performance.py` records memory truth, estimates, optional API latency, and real x264 vs VideoToolbox encodes; `benchmarks/` is gitignored and CI uploads the artifact | Fixed text→video/image prompts at 3 resolutions, quality frames, SSIM/PSNR/VMAF pair (needs loaded checkpoints and a dedicated run) |
| P1 | MPS safetensors `backend="pread"`; prompt-rewrite LRU cache keyed by settings revision/prompt/seed/image signature; Draft 8 (CFG off) · Standard 20 · Final 40 presets | True distilled 8+4 two-stage schedule (upstream pipeline exposes one `num_inference_steps`); embedding-level cache; gradient-estimating preset evaluation |
| P2 | Truthful process cap, allocator current/cache, per-stage sampler; frame-aware MPS pixel budget replacing the static 1280×768 cap; VAE tiling + `empty_cache` at chunk boundaries; measured Docker/Comfy child budget | VAE tile quality pair, torch 2.10 float32-VAE re-test, MLX spike, 481-frame validation |
| P3 | `MILIMO_RENDER_ENCODER=auto\|software\|videotoolbox` with CRF→`-q:v` mapping, `MILIMO_RENDER_CODEC=h264\|hevc`, x264 thread cap, real encode test; 720p review proxies generated on upload and exposed through `/uploads` + the media viewer | VMAF gate and default-auto promotion, `-hwaccel` input path, CPU-under-load measurement |
| P4 | Route-level lazy surfaces + lazy legacy editor; initial gzip 134.8 KiB studio shell / 173.3 KiB with Story (budget 250); RAF only while playing, cached clips/track state, single paused tick per seek; visibility-aware polling; sanitized + debounced persistence | React Compiler pilot, explicit 500-card/timeline recompute profiling, LUT 60fps path |
| P5 | WAL + `synchronous=NORMAL` + `busy_timeout` with a 3.45.1 caveat; JSON gzip; 1s cached `/health` plus cheap `/health/live`; upload thumbnails/proxies out of the `/uploads` request | SQLite 3.51.3 packaging, `/uploads` 100+ asset p95 measurement, thumbnail cache-invalidation test |
| P6 | `check:bundle` now walks the Vite manifest and enforces a 250 KiB initial-gzip route budget; CI runs the memory-truth gate, the agent-memory eval, and uploads the performance artifact | Nightly generation smoke and render-throughput/VMAF floors |

CLI: `./milimov/bin/python backend/check_memory_truth.py` and
`./milimov/bin/python backend/agent_memory_eval.py` are the two new hard gates.

## Findings → phases

| Phase | Findings | Theme |
|---|---|---|
| **P0** | PERF-02, PERF-20, PERF-22 | Measurement lab and budgets |
| **P1** | PERF-01, PERF-03, PERF-18 | Distilled schedule, loader, prompt cache |
| **P2** | PERF-04, PERF-05, PERF-08 | Memory, VAE, MLX lane, machine tiers |
| **P3** | PERF-06, PERF-07, PERF-19 | Render, VideoToolbox, proxies |
| **P4** | PERF-09, PERF-10, PERF-11, PERF-13, PERF-16 | Frontend runtime |
| **P5** | PERF-12, PERF-14, PERF-15, PERF-17 | Storage/API/thumbnails |
| **P6** | all | CI performance gates |

## P0 — Measurement lab (P1, 3–5 days)

No optimization without a baseline.

| Work | Acceptance |
|---|---|
| Benchmark harness | One command runs fixed text→video and image prompts at 3 resolutions, records load, stage 1, stage 2, VAE, encode, peak RSS/MPS, wall, and quality frames |
| Machine tag | Results include chip, RAM, macOS, torch, diffusers, FFmpeg, model hashes |
| Quality gate | SSIM/PSNR/LPIPS/VMAF pair against a reference frame/clip; never compare mp4 bytes (VideoToolbox keyframe placement varies) |
| Store artifact | `benchmarks/performance-<date>.json` + a markdown summary; gitignored by default, attach to releases |
| API/UI budget | Record 500-shot GET/PUT, `/health`, `/uploads`, board pan, timeline scroll, initial JS gzip |
| Baseline report | Current audit numbers reproduced on this machine before changing code |

**Dependencies:** none.

## P1 — Distilled schedule, loader, prompt cache (P1, 2–3 weeks)

| Work | Acceptance |
|---|---|
| Draft preset | Distilled `8 + 4` sigma schedule with no CFG for previews; one-step/second-stage preview option; UI labels Draft vs Final |
| Final preset | Keep print quality on the full/CFG path; document the quality delta with SSIM/VMAF |
| Gradient estimation | Evaluate `gradient_estimating_euler_denoising_loop` at 20–30 steps as an intermediate preset |
| Weight loader | Adopt upstream v1.6.0 `preadv` loader (or backport `backend="pread"` with a safetensors>=0.8 floor) on MPS; benchmark load and full run |
| Prompt embedding cache | Cache Gemma text embeddings keyed by prompt+negative+seed/resolution; measure the 8–9s-class saving |
| Residency | Keep the transformer resident between renders; free the text tower after encode; respect memory admission |
| Tests | Same-seed draft/final frames, cache hit/miss, loader fallback, peak/wall record |

**Dependencies:** P0.

## P2 — Memory, VAE, MLX lane, machine tiers (P1, 2–4 weeks)

| Work | Acceptance |
|---|---|
| VAE decode | Spatial tiles (e.g. 40 frames × 512 px, overlaps) for MPS/LTX-2.5; benchmark peak and VMAF/SSIM cost |
| Allocator/cache | Bound the MPS/MLX allocator during decode; test `torch.mps.empty_cache` timing and avoid thrash |
| Float32 VAE workaround | Re-test black-output fix on torch 2.10; if fp16/bf16 is safe, remove the fp32 penalty; otherwise document why |
| Residency policy | Keep transformer / unload text tower; measure load vs retain on 128GB |
| Machine tiers | Replace static 1280×768 cap with a measured ladder (RAM, chip, resolution, steps, estimated wall/peak) |
| Truthful cap | Reported ceiling equals the real `set_per_process_memory_fraction` cap; M0 owns the test |
| MLX spike | Optional LTX-2.5 q8 engine (Rapid-MLX/ltx-2-mlx) behind an engine flag; target 768×512×121 at ≤30 GB peak on 64 GB; compare wall/quality |
| Long clips | Validate 481-frame memory behavior; document limits |
| Tests | Peak/wall for each tier, VAE quality pair, MLX engine smoke |

**Dependencies:** P0, P1.

## P3 — Render, VideoToolbox, proxies (P1, 1–2 weeks)

| Work | Acceptance |
|---|---|
| Encoder choice | `MILIMO_RENDER_ENCODER=auto|software|videotoolbox`; H.264 and HEVC presets; `-q:v`/bitrate mapping |
| Quality gate | VMAF/SSIM against software output on reference cuts; default auto only when the threshold is met |
| CPU budget | Verify x264 thread count/nice and VideoToolbox CPU use under a concurrent UI/API load |
| Hardware decode | `-hwaccel videotoolbox` input path where supported; fall back cleanly |
| Proxy lane | Generate 720p H.264/HEVC proxies on ingest/job completion; Edit/Library use proxies, export uses originals |
| Cache | Reuse proxies/thumbnails; invalidate on replacement |
| Tests | Real ffmpeg render, VMAF threshold, proxy selection, missing-proxy fallback |

**Dependencies:** P0.

## P4 — Frontend runtime (P1, 1–2 weeks)

| Work | Acceptance |
|---|---|
| Lazy surfaces | Route-level lazy imports for Edit/Generate/Review/Library/Deliver/Shorts/Setup, Inspector, Comfy, LLM settings; initial gzip ≤250 KiB |
| PlaybackEngine | Cache audio clips/track state outside RAF; stop RAF when paused; resume on play; keep clock accuracy |
| Polling | One poller per resource, pause on `document.hidden`, refresh on visibility/resync; test request counts |
| React Compiler | Incremental pilot on one surface; compare board pan/timeline/initial render, keep manual memo |
| LUT preview | Keep 30fps cap; consider single WebGL/OffscreenCanvas path if 60fps is required |
| Lists | Audit 500-card/timeline recompute keys; no regression |
| Tests | Initial bundle budget, request-count test, paused-RAF CPU check, perf specs |

**Dependencies:** P0 for measurements.

## P5 — Storage, API, thumbnails (P2, 3–5 days)

| Work | Acceptance |
|---|---|
| SQLite upgrade | Use ≥3.51.3 (WAL-reset fix); document platform/packaging |
| WAL | `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout`, one writer; crash/recovery test |
| Compression | GZip/Brotli response middleware for JSON above a threshold |
| Health | Short-TTL cached memory snapshot + cheap liveness endpoint |
| Thumbnails | Generate posters on upload/job terminal, not inside `/uploads`; cache and invalidate |
| Uploads query | Measure 100+ assets; p95 budget |
| Tests | WAL concurrency, busy timeout, compression, cold/warm thumbnail timing |

**Dependencies:** P0.

## P6 — CI performance gates (P1, 3–5 days, continuous)

| Work | Acceptance |
|---|---|
| UI budgets | 500-card pan ≥50fps, 200-clip scroll ≥100fps, initial gzip ≤250 KiB |
| API budgets | 500-shot GET p95 ≤50ms, PUT p95 ≤150ms locally |
| Render budget | 10s 1080p encode throughput + VMAF floor in a nightly/manual job |
| Generation smoke | Tiny fixed model run nightly or on model changes; store wall/peak; fail only on regression thresholds, not absolute hardware |
| Artifact | Publish benchmark JSON/markdown per release |
| Docs | Update this plan and the audit with measured deltas |

**Dependencies:** P0; runs continuously.

## Sequencing

```text
P0 measurement
  ├─ P1 schedule/loader/prompt cache
  │    └─ P2 memory/VAE/MLX/tiers
  ├─ P3 render/proxies
  ├─ P4 frontend
  └─ P5 storage/API
P6 gates from P0 onward
```

## Effort

| Phase | Effort | Ships independently |
|---|---|---|
| P0 | 3–5 days | Yes |
| P1 | 2–3 weeks | Yes |
| P2 | 2–4 weeks | After P1 |
| P3 | 1–2 weeks | Yes |
| P4 | 1–2 weeks | Yes |
| P5 | 3–5 days | Yes |
| P6 | 3–5 days | Continuous |

Total: roughly **6–8 working weeks** for one engineer; P3/P4/P5 can run in
parallel with model work.

## Explicitly not in this plan

- A second full generation engine rewrite; MLX is a measured spike/optional lane.
- Sparse/windowed attention or int4 as primary levers.
- A new database engine; SQLite remains correct for a single-user local app.
- Cloud rendering or offloading generation.
- Changing output quality without a quality gate.

## Definition of done

- A repeated generation has recorded load/stage/VAE/peak/wall and a quality pair.
- Draft generation uses the distilled schedule; the measured speedup and quality
  delta are documented.
- MPS loading uses the `preadv`/pread path and is measurably faster.
- VideoToolbox is available behind a quality-gated preset; software remains for
  archival.
- Proxies make review playback smooth at 4K source sizes.
- Initial JS is split; paused Edit no longer runs an O(n) RAF; polling pauses
  when hidden.
- SQLite runs WAL on a fixed version with tested recovery.
- CI publishes benchmark budgets and blocks regressions.
