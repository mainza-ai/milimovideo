---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: memory management agentic mps mlx unified memory context kv cache audit
sources:
  - Local live memory snapshot, M3 Max 128 GB, macOS 27.3
  - https://docs.pytorch.org/docs/2.15/mps_environment_variables.html
  - https://docs.pytorch.org/docs/2.15/notes/mps.html
  - https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
  - https://ml-explore.github.io/mlx/build/html/python/memory_management.html
  - https://github.com/coeyai/ltx-2-mlx-speed-study/blob/main/STUDY.md
  - https://mem0.ai/blog/state-of-ai-agent-memory-2026
  - https://www.letta.com/blog/benchmarking-ai-agent-memory/
  - https://anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - https://docs.vllm.ai/en/stable/design/paged_attention/
  - https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html
  - https://github.com/bozbuilds/AIngram
related:
  - ../roadmap/memory-management-plan-2026-11-18.md
  - ../roadmap/performance-plan-2026-11-18.md
  - performance-audit-2026-11-18.md
  - memory-management-audit-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
---

# Memory Management Audit — 2026-11-18

## Verdict

The **GPU residency plane is strong**: measured admission, slot conflicts, leases,
TTL eviction, pressure signals, external-engine reporting, queue priority, OOM
retry, and per-job sampling exist and are tested. The gaps are now **truth,
runtime enforcement, calibration, and agentic memory**:

1. **The reported ceiling is 17.3 GB above the real process cap.** The snapshot
   reports `recommendedMaxWorkingSetSize` = 115.45 GB, while
   `set_per_process_memory_fraction(0.85)` caps the process at **98.13 GB**.
   Admission can approve a job the allocator must OOM.
2. **Memory is observed, not enforced, during a job.** The monitor runs every
   15s and only evicts idle models; an active job can walk into swap/jetsam while
   the sampler records it.
3. **Costs are static estimates.** Default video is budgeted at **95.55 GB**; an
   external M3 Max MLX study renders the same class of clip at **26 GB** after
   spatial VAE tiles and cache bounding. The app has no measured profile.
4. **Agentic memory is effectively absent.** The planner is deterministic, there
   is no long-term store or retrieval, and the 50-plan history is in-memory only.
   Context overflow is a hard error, not compaction.
5. **The browser repeats the whole document for every runtime update.** A 500-shot
   project is 203 KiB JSON; 20 progress edits measured **1.86 ms each** through
   persist + zundo + signature serialization.

## Live snapshot (this machine)

```text
total            137.44 GB
reserve           20.62 GB
reported ceiling 115.45 GB
real process cap  98.13 GB   (0.85 × recommendedMaxWorkingSetSize)
physical free     67.90 GB
swap used/free   23052 / 1596 MB (93.8%)
kernel pressure   normal
video estimate    95.55 GB   (121f 768×512)
image estimate    43.05 GB
```

Admission today:

```text
image: ok (43.05 GB)
video: refused — needs 95.5 GB (+4.8 margin) but only 67.9 GB RAM free
```

This is the correct refusal for the current machine state, but it also exposes
the mismatch: with 100 GB free the budget would admit 95.55 GB against a 98.13 GB
hard cap, leaving almost no allocator headroom.

## What already works

- Measured budget (`services/memory/budget.py`) with physical/RAM, MPS, CUDA,
  external engines, SAM, and swap signals.
- Admission with reclaim, retryability, downgrade ladder, safety margins, and
  actionable reasons.
- Slot conflict rules, leases, keep-warm, TTL eviction, forced eviction on
  pressure, and manual unload.
- External engine adapters for Ollama/oMLX, declared/unknown fallbacks, and
  unload.
- Durable queue priority/same-slot grouping, bounded memory waits, one OOM retry,
  and a job-event `memory` trace.
- Per-job sampler peaks by stage, persisted to `memory_json`.
- 232 backend tests including budget/admission/lease/eviction/engines and memory
  endpoints.

## Research highlights

| Source | Date | What changes the plan |
|---|---|---|
| [PyTorch MPS env vars](https://docs.pytorch.org/docs/2.15/mps_environment_variables.html) | Sep 2026 | High watermark default **1.7**, low **1.4**; high is a hard allocation limit, low is adaptive-commit/GC. Fraction is relative to `recommendedMaxWorkingSetSize`. |
| [PyTorch MPS notes](https://docs.pytorch.org/docs/2.15/notes/mps.html) | Oct 2026 | No float64; no blanket performance guarantee; fallback cannot rescue double. |
| [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) | 2026 | CPU/GPU share one pool; no device copies; schedule ops on the best unit. |
| [MLX memory API](https://ml-explore.github.io/mlx/build/html/python/memory_management.html) | 2026 | `get_active_memory`, `get_peak_memory`, `get_cache_memory`, `set_memory_limit`, `set_cache_limit`, `set_wired_limit`, `clear_cache` — the controls this app does not use. |
| [M3 Max MLX study](https://github.com/coeyai/ltx-2-mlx-speed-study/blob/main/STUDY.md) | Sep 2026 | Decode peak 53.4→26.1 GB with spatial tiles; cache limit 0 cuts 44.2→25.8 GB; keep transformer, free text tower; prompt embedding cache saves 8–9s. |
| [Mem0 state of agent memory](https://mem0.ai/blog/state-of-ai-agent-memory-2026) | Oct 2026 | Separate working/episodic/semantic/procedural memory; multi-scope IDs; multi-signal retrieval; privacy/staleness/security remain open. |
| [Letta memory benchmark](https://www.letta.com/blog/benchmarking-ai-agent-memory/) | 2025/2026 | A filesystem plus good tools reached 74% on LoCoMo; agent capability and context management can matter more than the store. |
| [Anthropic context engineering](https://anthropic.com/engineering/effective-context-engineering-for-ai-agents) | 2025/2026 | Context rot is real; compaction, structured notes, just-in-time retrieval, and sub-agents are the practical long-horizon tools. |
| [vLLM PagedAttention](https://docs.vllm.ai/en/stable/design/paged_attention/) | 2026 | Block-paged KV cache and prefix sharing; relevant if the app ever hosts its own in-process LLM serving. |
| [OWASP agent security](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html) | 2026 | Memory poisoning is a named agentic risk; memory needs provenance, validation, and rollback. |
| [sqlite-vec/local memory](https://github.com/bozbuilds/AIngram) | 2026 | Local SQLite + vector/FTS hybrid memory is practical and fast at 10K–100K entries. |

## System memory findings

### MEM-01 — Reported ceiling exceeds the real MPS cap (P1)

`budget.snapshot()` sets `ceiling = min(recommended_max_memory, total - reserve)`
(`budget.py:115-120`), while `set_mps_cap()` calls
`torch.mps.set_per_process_memory_fraction(0.85)` (`budget.py:458-473`).
The fraction applies **to** `recommendedMaxWorkingSetSize`:

```text
reported ceiling 115.45 GB
process cap      115.45 × 0.85 = 98.13 GB
gap              17.32 GB
```

The admission decision, Setup panel, and API all report 115.45 GB. Fix: report
both `device_cap_gb` and `budget_ceiling_gb`, and use the process cap in
`available`/`_fits`.

### MEM-02 — MPS allocator is not explicitly configured or observed (P1)

There is no `PYTORCH_MPS_HIGH_WATERMARK_RATIO`/`LOW_WATERMARK_RATIO` policy, no
`current_allocated_memory`/`driver_allocated`/`recommended` breakdown in the
sampler, and no allocator stats in the UI. Only `driver_allocated_memory()`
(`sampler.py:31-38`) is sampled once per second. The PyTorch defaults (1.7/1.4)
are unrelated to the 0.85 fraction, and adaptive commit behavior is opaque.

### MEM-03 — Cost registry is static and unmeasured (P1)

`VIDEO_BASE_GB=76`, `LTX25_BASE_GB=64`, `IMAGE_BASE_GB=36`, and workspace formulas
are hand-tuned (`costs.py`). The default video estimate is **95.55 GB**;
`benchmark_models.py` prints only estimates (measured peak/load/wall are blank).
The MLX study reaches **26 GB** peak for a comparable render after memory work.
Admission is therefore both over- and under-estimating depending on the lane.

### MEM-04 — No runtime enforcement while a job is active (P1)

The pressure monitor runs every **15s** (`MEMORY_MONITOR_INTERVAL_SEC`), and on
warning/critical it calls `evict_idle(force=True)` (`memory_manager.py:401-416`).
Leased slots are skipped, so an active job can keep allocating while the system
swaps. The sampler records the peak but cannot pause, cancel, or degrade the job.
macOS Jetsam can kill the process before Python sees an OOM.

### MEM-05 — Sampler has blind spots and no user-facing trace (P1/P2)

- 1s interval can miss sub-second peaks.
- Only MPS driver allocation or process RSS; no physical free, swap free,
  compressed/wired memory, or per-process RSS for ffmpeg/Comfy/SAM.
- No current/cache/peak distinction; no min/avg/trend.
- Per-job `memory_json` is persisted but not shown in Studio; `/metrics/jobs`
  exposes only the max across jobs. Refusal history is not surfaced beyond
  counters.

### MEM-06 — Full state dict and per-stage reload waste memory and time (P1)

The vendored loader builds a complete `dict` of tensors before the model consumes
it (`sft_loader.py:20-45`), and the two-stage pipeline builds/frees the transformer
per stage. Upstream measured **2.6x** end-to-end on a small MPS run by switching
to `preadv` and noted the full checkpoint is resident before consumption. This is
both a transient-memory and load-time problem.

### MEM-07 — No MLX allocator policy or lane (P1)

MLX exposes `set_memory_limit`, `set_cache_limit`, `set_wired_limit`,
`get_active_memory`, `get_cache_memory`, `get_peak_memory`, and `clear_cache`; the
app uses none. The M3 Max study shows the allocator free list can dominate the
decode peak and that a bounded cache cuts 44.2→25.8 GB with identical pixels.

### MEM-08 — External residency is partly blind (P1/P2)

- SAM is budgeted from a health boolean: 8 GB loaded / 0.5 GB idle
  (`budget.py:65-84`), not measured RSS.
- Comfy defaults to a declared 2 GB (`COMFY_DECLARED_GB`), regardless of the
  models it has cached.
- `MEMORY_UNKNOWN_ENGINE_GB=24` is defined but unused; unknown engines are
  estimated from model names.
- Strata cannot unload and is only budgeted from a declared override.

### MEM-09 — Child-process memory is a fixed guess (P1/P2)

`VIDEO_ENCODE_GB=2.0` is the only encode budget. x264 uses 24 threads on this
machine and PyAV holds frame buffers; a 4K/long clip or Comfy/SAM child can exceed
the slot estimate. There is no child RSS sampling, thread cap, or `nice` policy.

### MEM-10 — Browser document memory repeats full-project serialization (P2)

Zustand `persist` stores the whole project; zundo keeps 30 snapshots and runs a
`JSON.stringify` equality check; the save-status subscription hashes the whole
project. A 500-shot project is **203 KiB** JSON, and 20 runtime progress updates
measured **1.86 ms each** in the store probe. Progress events therefore block the
main thread and write runtime fields to localStorage that hydration throws away.

### MEM-11 — SQLite page cache/WAL is not part of the memory model (P2)

SQLite is in DELETE/FULL mode; page cache and WAL would consume process memory and
are not budgeted. The performance plan already covers WAL; the memory plan should
add a page-cache cap and include it in the process estimate.

### MEM-12 — Residency policy is global and static (P2)

TTLs are fixed (video/image 300s, LLM 120s, SAM 600s), keep-warm is a global env
default plus per-slot pin, and there is no per-project/session policy. A user who
alternates storyboard/prompt/model jobs pays reloads; a user who leaves a 95 GB
model resident blocks other work.

## Agentic memory findings

### MEM-13 — There is no long-term agent memory (P1)

`agentStore` holds `lastPlan` plus a **50-item in-memory history**, lost on reload.
The planner is deterministic and explicitly “never calls a model”
(`routes/planner.py:1-12`). There is no store for facts, decisions, feedback,
takes, or project history. The `agent_plan`, `review_critic`, and `continuity`
roles are configured but effectively unused.

### MEM-14 — Context overflow is a hard error, not compaction (P1)

`_context_guard` estimates tokens and raises when the prompt is >95% of the model
context (`runtime.py:150-170`). There is no token budget by section, no
summarization/compaction, no structured note-taking, and no just-in-time retrieval.
Anthropic’s context-engineering guidance and the context-rot evidence both point
to compaction/notes/sub-agents as the practical answer.

### MEM-15 — Project memory is not retrieved into prompts (P1)

Takes, review comments/approvals, board notes, references, and prior prompts live
in SQLite/local stores but are not retrieved for plan/continuity/review roles.
Search is lexical (`search/localSearch.ts` says “not an embedding model”). There is
no semantic, episodic, or procedural memory layer for the film.

### MEM-16 — No memory scopes, lifecycle, or governance (P2)

There is no `user/project/shot/session/agent` scope, no TTL/decay/consolidation/
dedupe, no staleness marker, no export/delete UI, and no retention policy. A future
memory store must be project-scoped and deletable with the project.

### MEM-17 — No memory security model (P1/P2)

OWASP names memory/context poisoning (ASI06) as an agent risk. Imported scripts,
comments, and notes can contain instructions; there is no provenance tag, trust
level, signed entry, validation, quarantine, or rollback. A local app is not
immune: the user may import third-party content.

### MEM-18 — No memory evaluation (P1)

There is no app-specific memory benchmark. Public numbers are contested: Mem0
self-reports 92.5 LoCoMo/94.4 LongMemEval, Letta reports a filesystem approach at
74% and independent runs differ. The only meaningful gate is project tasks such
as “find the approved take for shot 7,” “what did the reviewer reject,” or
“continue this scene with the same character.”

### MEM-19 — Prompt embeddings are recomputed (P2)

The text tower is released after encode and no embedding cache exists. The MLX
study saves 8–9s per repeated prompt with a small embedding cache; this also
reduces repeated model residency churn.

### MEM-20 — Agentic memory observability is absent (P2)

There is no memory panel showing what the agent remembers, why it was retrieved,
token cost, or which entries were used. Without that, staleness, poisoning, and
privacy cannot be audited by the user.

## Missing functionality / enhancements

1. **Truthful memory accounting**: reported cap = process cap; separate device
   cap, budget, physical free, swap free, compressed/wired; per-process and
   child-process RSS.
2. **Runtime guard**: a pressure loop that can pause admission, evict, cancel, or
   downgrade an active job before Jetsam.
3. **Measured profiles**: per-model/per-resolution load, peak, stage peaks, wall;
   replace static estimates over time.
4. **Allocator policy**: MPS high/low watermarks, cache stats, `empty_cache`
   discipline; MLX memory/cache limits for an optional lane.
5. **Streaming loader**: preadv/pread weight loading, avoid the full state dict,
   reuse the transformer between stages.
6. **Browser memory discipline**: sanitized/debounced persistence, IndexedDB,
   capped undo, media buffer cleanup, memory panel for the webview.
7. **Agent memory service**: SQLite + FTS/vector store with project/shot/session
   scopes, working/episodic/semantic/procedural records, provenance, TTL/decay,
   consolidation, deletion.
8. **Context packer**: token budget, retrieval, compaction/summarization,
   structured notes, tool-result clearing, “what was used” trace.
9. **Memory security**: trust levels, validation/quarantine, signed entries,
   rollback, deletion audit (OWASP ASI06).
10. **Memory evaluation**: project-specific recall/task benchmarks plus token and
    latency budgets.

## Recommended direction

- **Make the number honest before optimizing it.** A 17 GB overstatement is worse
  than a conservative budget.
- **Enforce, don’t just observe.** The sampler should feed a control loop that can
  act within seconds.
- **Measure each lane.** Static costs cannot cover LTX-2, LTX-2.5, Wan, Flux,
  SAM, Comfy, and MLX.
- **Treat memory as a hierarchy, not one pool.** Weights, activations, VAE decode,
  encoder, KV/embeddings, browser documents, and agent memory have different
  lifetimes and controls.
- **Adopt agent memory deliberately.** Start with project-scoped notes/decisions
  and retrieval; do not add a vector database before an app-specific eval exists.
- **Use local SQLite for memory.** It matches local-first, supports FTS and
  sqlite-vec, and can be exported/deleted with the project.

## Implementation plan

The memory plan (M0–M5) and its relationship to performance P2:
[[../roadmap/memory-management-plan-2026-11-18.md]].

## Implementation status (2026-11-18)

Resolved on this branch: separate device-recommended/process-cap/ceiling values
with admission on the cap (98.13 GB here), allocator current/cache, 0.5s
per-stage sampler with physical/swap minima, job traces in `/system/memory` and
MemoryPanel, pressure guard with block/evict/cancel actions, child-process RSS,
sanitized/debounced persistence, paused-RAF playback, hidden-tab polling, and a
project-scoped agent memory store with FTS5 retrieval, token budgets, provenance,
quarantine/approve, export/delete, planner RAG, and a 20-task eval (top-1/top-3 100%,
p95 ~1.6 ms).

Still open: per-lane measured model profiles and VAE quality pairs, MLX cache
controls, browser IndexedDB migration and media-buffer audit, consolidation and take
extraction, and a larger evaluation suite. See
[[../roadmap/memory-management-plan-2026-11-18]].
