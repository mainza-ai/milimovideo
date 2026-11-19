---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: performance memory implementation benchmark mps videotoolbox sqlite react agentic
sources:
  - ../roadmap/performance-plan-2026-11-18.md
  - ../roadmap/memory-management-plan-2026-11-18.md
  - performance-audit-2026-11-18.md
  - memory-management-audit-2026-11-18.md
  - ../../backend/check_memory_truth.py
  - ../../backend/agent_memory_eval.py
  - ../../backend/benchmark_performance.py
related:
  - ../roadmap/performance-plan-2026-11-18.md
  - ../roadmap/memory-management-plan-2026-11-18.md
  - agentic-memory.md
  - ../concepts/memory-slots.md
  - architecture.md
---

# Performance and Memory Execution (2026-11-18)

Implementation record for the performance P0–P6 and memory M0–M5 plans. This page
is the evidence map: what shipped, what it measures, and what remains open. The
plans keep the acceptance criteria; this page records the delivered state.

## Phase map

| Phase | Delivered | Evidence |
|---|---|---|
| **P0** benchmark lab | `backend/benchmark_performance.py` records memory truth, estimates, optional API latency, and real x264 vs VideoToolbox encodes; `benchmarks/` is gitignored and CI uploads the artifact | local run below; CI artifact |
| **M0** truth | `device_recommended_gb`, `process_cap_gb`, allocator current/cache; admission on the cap; 0.5 s sampler; job traces and recent jobs in `/system/memory`; MemoryPanel cap/allocator/sampler UI | `check_memory_truth.py`, `tests/test_memory_budget.py`, `tests/test_storage_and_health.py` |
| **P1** generation | MPS safetensors `backend="pread"`; prompt-rewrite LRU cache keyed by settings revision/prompt/seed/image; Draft 8 / Standard 20 / Final 40 | `tests/test_safetensors_loader.py`, `tests/test_prompt_cache.py` |
| **M1** calibration/loader | `pread` loader, transformer reuse, VAE tiling + `empty_cache` boundaries, prompt cache | loader/cache tests; model-load profiles still open |
| **P2** memory/tiers | frame-aware MPS pixel budget replacing static 1280×768; truthful cap; child process truth | `tests/test_mps_resolution.py`, `tests/test_memory_guard.py` |
| **M2** runtime guard | warning blocks admission and evicts idle; critical marks the oldest active job for cancellation with a reason; `memory.pressure`; child RSS/threads | `tests/test_memory_guard.py` |
| **P3** render/proxy | `MILIMO_RENDER_ENCODER=auto\|software\|videotoolbox`, `MILIMO_RENDER_CODEC=h264\|hevc`, CRF→`-q:v`, x264 thread cap, real encode test; background 720p upload proxies exposed in `/uploads` and Library previews | `tests/test_render_encoder.py`, `tests/test_media_thumbnails.py` |
| **P4** frontend | lazy route surfaces + lazy legacy editor; initial studio gzip 134.8 KiB (173.3 with Story); rAF only while playing; visibility-aware polling | `tests`/e2e, `check:bundle` |
| **M3** browser | sanitized + 400 ms debounced localStorage, runtime fields stripped, browser/document/undo/localStorage stats in MemoryPanel | `web-app/src/studio/state/documentStore.test.ts` |
| **P5** storage/API | WAL + `synchronous=NORMAL` + 5 s busy timeout; JSON gzip; 1 s cached `/health` + `/health/live`; `/uploads` no longer runs ffmpeg | `tests/test_storage_and_health.py` |
| **M4** agentic memory | project-scoped SQLite + FTS5/LIKE, kinds/scopes, provenance/trust, dedupe, token budget, quarantine/approve, export/clear, review extraction, planner RAG, Setup panel | `tests/test_agent_memory.py`, [[agentic-memory]] |
| **M5** eval/security | 20-task eval, poisoning fixtures, deletion/export audit, CI gates | `agent_memory_eval.py` |
| **P6** gates | manifest-walked 250 KiB initial-gzip budget; CI memory-truth gate; retrieval eval; performance artifact upload | `.github/workflows/ci.yml`, `web-app/scripts/check-bundle.mjs` |

## Measurements on the reference machine

M3 Max, 128 GB, macOS 27.3, torch 2.10, safetensors 0.8, FFmpeg 9.0.2:

- Device recommended **115.45 GB**, real process cap **98.13 GB**; admission and
  the reported ceiling both use 98.13 GB.
- Physical free ~66–68 GB; swap free ~1.6 GB (swap ~94% occupied, pressure
  `normal`); the budget reports swap but does not refuse on historical occupancy.
- Cost estimates: video **95.55 GB** (76 base + 15 workspace) is refused; image
  **43.05 GB** is admitted.
- Real 5 s 1080p encode: x264 **0.506 s / 6.13 MB**; VideoToolbox **0.621 s /
  5.38 MB** (keyframe placement and size differ; software remains archival).
- Agent memory eval: 20 tasks, top-1/top-3 **100%**, ~199 tokens/pack, p50 ~0.8 ms,
  p95 ~1.6 ms.
- Frontend initial gzip: entry 77.0 KiB, studio shell **134.8 KiB**, studio + Story
  **173.3 KiB**, legacy **177.2 KiB**.

## Verification

- Backend: `compileall`, `ruff check backend`, **270 pytest passed**.
- Frontend: `tsc -b`, ESLint 0 errors, **118 Vitest passed**, Vite build,
  `check:bundle`.
- E2E: **32 Playwright passed**.
- Environment: `verify_setup.py` 0 failures / 0 warnings.
- CI: backend + frontend green on `main` and the feature branch.

## Still open

- Fixed-prompt model benchmarks at 3 resolutions with quality frames (SSIM/PSNR/
  VMAF), per-lane measured profiles within ±15%, VAE tile quality pair.
- A VMAF gate before `auto` may choose VideoToolbox; hardware-decode input path.
- MLX lane, 481-frame validation, torch 2.10 float32-VAE re-test.
- React Compiler pilot, IndexedDB migration for very large documents, media-buffer
  audit.
- SQLite 3.51.3 packaging (WAL-reset fix), `/uploads` 100+ asset p95 measurement.
- Agent memory consolidation, automatic take extraction, larger eval suite.

## Related

- [[../roadmap/performance-plan-2026-11-18.md]] — P0–P6 acceptance criteria
- [[../roadmap/memory-management-plan-2026-11-18.md]] — M0–M5 acceptance criteria
- [[agentic-memory.md]] — the memory store in detail
- [[../concepts/memory-slots.md]] — slots and admission
- [[architecture.md]] — module map
