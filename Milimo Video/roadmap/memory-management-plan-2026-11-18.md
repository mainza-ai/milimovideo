---
type: roadmap
created: 2026-11-18
updated: 2026-11-18
tags: memory plan unified memory mps mlx agentic context kv cache roadmap
sources:
  - ../systems/memory-management-audit-2026-11-18.md
  - ../roadmap/performance-plan-2026-11-18.md
  - ../systems/memory-management-audit-2026-11.md
related:
  - ../systems/memory-management-audit-2026-11-18.md
  - performance-plan-2026-11-18.md
  - implementation-plan-2026-11.md
  - open-work-resolution-2026-11.md
---

# Memory Management Plan (2026-11-18)

Input: [[../systems/memory-management-audit-2026-11-18.md]] — 20 findings
(`MEM-01…MEM-20`) covering system/GPU memory, child processes, browser memory, and
agentic memory.

This plan **extends** [[performance-plan-2026-11-18]]:

- **P0** owns the benchmark harness; M0 consumes it for memory truth.
- **P2** owns VAE/MLX/machine tiers; M1/M2 deepen the allocator, residency,
  loader, and runtime guard.
- **P3** owns render; M2 adds child-process budgets.
- **P4** owns frontend runtime; M3 adds browser/Tauri memory discipline.
- **M4** is new: agentic memory and context engineering.
- **P6** owns performance CI; M5 adds memory evaluation/security gates.

Priority: **truth → allocator/loader → runtime guard → browser → agentic memory →
evaluation/security**.

Status: **M0, M2, M3, M4 core, and M5 gates implemented 2026-11-18;
measured model profiles, VAE quality pairs, and the MLX lane remain open.**

## Implementation status (2026-11-18)

| Phase | Landed | Still open / deliberately deferred |
|---|---|---|
| M0 | `device_recommended_gb`, `process_cap_gb`, allocator current/cache; admission uses the real cap (98.13 GB on this 115.45 GB-recommended M3 Max); 0.5s sampler with stage peaks, physical minimum, swap peak; `/system/memory` job traces + children; MemoryPanel shows cap, allocator, sampler, recent peaks | Cost-registry `measured|estimated|declared` confidence labels per lane |
| M1 | `pread` loader; transformer reuse path; VAE tiling; frame-aware resolution budget; prompt-rewrite cache | Per-lane measured profiles (±15%), VAE tile quality pair, MLX engine, residency policy UI |
| M2 | 1–2s pressure guard: warning blocks admission + evicts idle; critical marks the oldest job for cancellation with a reason; `memory.pressure` event; child-process RSS/thread truth; external engines already budgeted | Reservation model for browser/VM/other apps; fake-pressure integration test through the running server |
| M3 | Sanitized + debounced localStorage (runtime fields stripped), PlaybackEngine runs only while playing, visibility-aware polling, browser/document/undo/localStorage statistics in MemoryPanel | IndexedDB migration for very large documents, media buffer release audit, waveform cache cap |
| M4 | Project-scoped SQLite memory with FTS5 (LIKE fallback), kinds/scopes, provenance/trust, dedupe, token-budgeted context pack with "why", quarantine + approve, export/clear, planner RAG, Setup panel | Automatic extraction from takes, consolidation/summarization, sqlite-vec hybrid |
| M5 | 20-task retrieval eval (top-1/top-3 100%, ~199 tokens, p95 ~1.6 ms), poisoning fixtures, deletion/export audit, CI gates, memory store size is bounded by per-entry and budget limits | Larger 50-task suite, per-request token deny policy, store-size quota |

Security evidence: `tests/test_agent_memory.py` rejects agent/imported text that
matches instruction-injection signatures, keeps it out of context packs, and
only admits it after explicit approval.

## Findings → phases

| Phase | Findings | Theme |
|---|---|---|
| **M0** | MEM-01, MEM-02, MEM-05, MEM-19 | Truth, allocator stats, observability |
| **M1** | MEM-03, MEM-06, MEM-07, MEM-12, MEM-19 | Calibration, loader, residency, MLX |
| **M2** | MEM-04, MEM-08, MEM-09 | Runtime guard, children, external engines |
| **M3** | MEM-10, MEM-11 | Browser/document/SQLite memory |
| **M4** | MEM-13…MEM-18, MEM-20 | Agentic memory and context |
| **M5** | all | Evaluation, security, CI gates |

## M0 — Truth and observability (P1, 3–4 days)

| Work | Acceptance |
|---|---|
| Effective cap | Report `device_recommended_gb`, `process_cap_gb` (recommended × fraction), `budget_ceiling_gb`, `physical_free_gb`, `swap_free_gb` separately; admission uses the *minimum* |
| Allocator stats | Expose MPS `driver_allocated`, `current_allocated`, `recommended_max`, allocator cache/peak; CUDA equivalent; document high/low watermark |
| Sampler | 250ms–1s cadence; record min/avg/peak by stage; system pressure/swap/physical; external/child RSS where available |
| Per-job trace | Studio shows `memory_json` per job: stage peaks, swap, samples, estimate vs actual; `/metrics/jobs` keeps aggregate |
| Cost truth | Show estimate source/confidence (`measured|estimated|declared`) and the last measured profile for the lane |
| Tests | Ceiling equals the actual cap; sampler fake-clock; trace rendering; preflight reason |

**Dependencies:** P0 harness.

## M1 — Calibration, loader, residency, MLX (P1, 1–2 weeks)

| Work | Acceptance |
|---|---|
| Measured profiles | Per lane (LTX-2, LTX-2.5, Wan, Flux, SAM, Comfy) record load/peak/stage peak/wall at 2–3 resolutions; estimates within ±15% or marked unknown |
| Streaming loader | Adopt v1.6.0 `preadv`/pread (or `backend="pread"` with safetensors≥0.8); avoid the full state dict; measure transient RSS/MPS peak |
| Transformer reuse | Keep the transformer between stages/renders where memory allows; free the text tower; test two-stage peak |
| VAE/decode | Spatial tiles + bounded allocator/cache; measure peak and SSIM/VMAF cost |
| MPS policy | Set explicit high/low watermarks around the per-process cap; `empty_cache` only at safe boundaries; test thrash |
| MLX lane | Optional MLX engine with `set_memory_limit`/`set_cache_limit`/`set_wired_limit`, peak/active/cache telemetry; target 768×512×121 ≤30 GB on 64 GB |
| Residency policy | Per-project/session TTL/keep-warm; adaptive by lane and last use; expose in Setup |
| Prompt cache | Cache text embeddings keyed by prompt/negative/seed/resolution; measure 8–9s-class saving |
| Tests | Loader memory/time, transformer reuse peak, VAE tile quality, MLX cache, residency policy |

**Dependencies:** M0, P0.

## M2 — Runtime guard, children, external engines (P1, 1 week)

| Work | Acceptance |
|---|---|
| Pressure loop | 1–2s cadence; warning/critical thresholds; actions: block new admission, evict idle, pause queue, cancel or downgrade active job |
| Pre-Jetsam safety | No active job crosses critical pressure; emit `memory.pressure`/`job.degraded`/`job.cancelled` with reason; never kill the backend silently |
| Child budgets | ffmpeg/PyAV/Comfy/SAM child RSS and threads included in the estimate; cap x264 threads and priority; measure 4K/long-clip encode |
| External engines | SAM actual RSS/health, Comfy cached model size, unknown-engine fallback; unload when safe |
| Reservation | Reserve memory for UI/OS/encode; account for browser webview and VM/other apps |
| Tests | Fake pressure escalation, cancel/degrade path, child RSS estimate, external unknown |

**Dependencies:** M0, P0.

## M3 — Browser/Tauri memory discipline (P2, 3–5 days)

| Work | Acceptance |
|---|---|
| Persist | Persist a sanitized project (no runtime fields), debounce/throttle, move large documents to IndexedDB; localStorage only for small prefs |
| Undo | Cap/compact zundo snapshots by content; avoid full JSON equality per runtime update |
| Signature | Compute save signature only for document changes, not progress/status; skip on runtime-only updates |
| Media | Release video/audio/canvas buffers on close; cap waveform/peak caches; avoid duplicate media downloads |
| Memory panel | Show webview heap (if available), document JSON size, undo snapshots, localStorage/IndexedDB usage |
| Tests | 500-shot update cost <0.5ms, persist size sanitized, undo memory bound, media cleanup |

**Dependencies:** P4.

## M4 — Agentic memory (P1, 3–4 weeks)

Start with project-scoped notes/decisions, not a vector database.

| Work | Acceptance |
|---|---|
| Memory store | SQLite tables (project/shot/session/agent scope, kind, text, metadata, timestamps, provenance, trust, hash); optional FTS5 + sqlite-vec hybrid; no cloud |
| Memory types | Working (context pack), episodic (takes/reviews/attempts), semantic (story facts/characters/locations), procedural (workflow/preferences) |
| Write path | Explicit tools (`remember`, `forget`, `search`) + automatic extraction from approvals/comments/takes with provenance; dedupe/merge/conflict |
| Retrieval | Multi-signal (FTS + vector + entity + recency); token-budgeted packer; "why retrieved" trace |
| Context | Compaction/summarization near limit; structured notes; tool-result clearing; section token budgets; no hard-fail on long prompts |
| RAG | Retrieve project facts, approved takes, review feedback, board notes, references; propose before spending |
| Lifecycle | TTL/decay, staleness marker, consolidation, export/delete with project; scope isolation |
| Security | Trust levels for imported/user/agent content; validation/quarantine; rollback of poisoned entries; deletion audit (OWASP ASI06) |
| UI | Memory panel: browse/search/edit/delete, provenance, last used, token cost; clear-all/export |
| Tests | Scoped retrieval, conflict update, compaction, token budget, deletion/export, poisoning rejection, app-specific tasks |

**Dependencies:** P0 for token/latency measurement; M0 for memory truth.

## M5 — Evaluation, security, and gates (P1, 1 week, continuous)

| Work | Acceptance |
|---|---|
| App memory eval | 20–50 project tasks: find approved take, recall rejection reason, continue character, resolve changed fact; measure accuracy, tokens, latency |
| System benchmark | Per-lane load/peak/stage/wall; no regression beyond thresholds; publish artifact |
| Security | Memory-poisoning fixtures, provenance/trust checks, deletion/export audit |
| Budgets | Per-request token budget, retrieval top-k, memory store size; warn/deny over budget |
| CI | Memory tests + benchmark artifact; fail on cap mismatch, stale trace, or privacy deletion failure |

**Dependencies:** M0–M4.

## Sequencing

```text
M0 truth
 ├─ M1 allocator/loader/residency/MLX
 │    └─ M2 runtime guard/children
 ├─ M3 browser/document
 └─ M4 agentic memory
M5 evaluation/security from M0 onward
```

## Effort

| Phase | Effort | Ships independently |
|---|---|---|
| M0 | 3–4 days | Yes |
| M1 | 1–2 weeks | Yes |
| M2 | 1 week | After M0 |
| M3 | 3–5 days | Yes |
| M4 | 3–4 weeks | After M0 |
| M5 | 1 week | Continuous |

Total: roughly **6–8 working weeks**; M1/M2 overlap performance P2, M3 overlaps P4,
M4 runs in parallel after M0.

## Explicitly not in this plan

- No cloud memory or hosted vector database.
- No multi-user/tenant memory model; this is a local single-user studio.
- No in-process LLM serving rewrite; vLLM/PagedAttention is a reference, not a new
  backend.
- No memory optimization without a measured profile and quality gate.
- No removal of existing manual memoization before a React Compiler pilot.

## Definition of done

- The reported memory ceiling never exceeds the process cap; preflight shows both.
- Allocator and system stats are visible per job; peaks are not blind 1s samples.
- Loader and transformer reuse are measured; no full state-dict transient.
- A pressure spike can pause/cancel/degrade before Jetsam.
- A 500-shot project does not serialize the document on every progress event.
- Agent memory is project-scoped, retrievable, provenance-aware, deletable, and
  evaluated on real studio tasks.
- Context overflow is handled by compaction/retrieval, not a hard failure.
- Security tests reject poisoned memory and verify deletion.
