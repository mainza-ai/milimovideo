---
type: roadmap
created: 2026-11-17
updated: 2026-11-17
tags: open-work plan ci cuda packaging wan quantized cutover
sources:
  - ../../.github/workflows/ci.yml
  - ../../backend/requirements.txt
  - ../../backend/tasks/image.py
  - ../../backend/tasks/video.py
  - ../../LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py
related:
  - implementation-plan-2026-11.md
  - ../systems/github-issues-triage-2026-11.md
  - ../systems/memory-management-audit-2026-11.md
  - ../systems/job-notifications-audit-2026-11.md
---

# Open Work and Resolution Plan (2026-11-17)

A point-in-time inventory of everything still open after the memory/event
implementation, with an ordered resolution plan. "Done" means code on `main` plus
a test; "open" means no code or no verified path.

## 1. Inventory

### GitHub

| Item | State | Finding |
|---|---|---|
| Issue #6 (16 GB CUDA OOM) | Open | **Mitigated** by CUDA VRAM admission; CPU offload not implemented. Close after a maintainer reply. |
| Issue #4 (AE 404) | Open | **Fixed** in `c955ac0`; close with the repo/licence note. |
| Issue #3 (API) | Open | **Resolved** by `docs/07_api.md`; reply and close. |
| Issue #2 (quantized models) | Open | **Feature gap**: no GGUF loader or quantized Flux entry. |
| Issue #1 (Wan 2.2) | Open | **Feature gap**: a separate model family, not an LTX patch. |
| PR #7 (`enhance_prompt` tuple) | Open | **Superseded**: both call sites already unpack on `main` (`tasks/image.py:98`, `tasks/video.py:290`). Close with a pointer. |
| PR #8 (`encode_video` no_grad) | Open | **Still valid**: `media_io.py` has no `torch.no_grad()` around generator consumption; the CUDA inference-tensor crash is real. Rebase, test, merge. |
| CI on `main`/`develop` | **RED** | Backend `Tests` fails: 41 failed, 37 errors. Reproduced locally with SQLModel 0.0.50 + SQLAlchemy 2.1.9: `Datetime values must have timezone information`. Local pinned versions (SQLModel 0.0.31 / SQLAlchemy 2.0.46) pass 180 tests. |
| Security alerts | Disabled | `vulnerability-alerts` returns 404; Dependabot/security alerts are not enabled. |

### Roadmap / AGENTS deferrals

- Bundled Python/Torch sidecar (desktop packaging).
- Legacy `/` cutover (gated on two clean releases — currently impossible with red CI).
- Scoped-v1: transitions/LUT/titles/A2, embedding search, shorts/lens packs.
- LLM caveats: non-stream cancellation, chained chunk-0 enhancement, Strata
  context, rate-card overrides, heuristic vision, legacy popover.
- Memory: hand-tuned costs, Strata unload, cancelled pipeline TTL, MPS-only tiling,
  20 memory waits, CUDA offload and quantized checkpoints.
- M5 artifact work; Phase 6 v2 lane (eval harness, model swaps, MLX/FP8, crew).
- Per-job priority override; desktop tray icon; legacy notification surface.
- Visual baselines are local/platform-tolerant.
- One perf spec is flaky under load (500-card pan threshold).

## 2. P0 — Restore a trustworthy baseline (1–2 days)

**Goal:** green CI on `main` and `develop`, stale contributions resolved, GitHub
issues triaged. Everything else is blocked on this.

| Step | Change | Acceptance |
|---|---|---|
| P0-1 | Pin the known-good DB stack in `backend/requirements.txt`: `sqlmodel==0.0.31`, `sqlalchemy>=2.0.14,<2.1` (or migrate every model to `NaiveDatetime`/aware datetimes and keep 2.1). Add a CI `pip check`. | Backend CI green on both branches; local and CI resolve the same SQLAlchemy. |
| P0-2 | Apply PR #8 (wrap generator consumption in `torch.no_grad()`, `.detach()` before CPU) and add a CPU regression test that encodes a generator under `torch.inference_mode()`. | Test passes; manual CUDA checklist passes; PR merged with credit. |
| P0-3 | Close PR #7 as superseded with line references. | PR closed; no duplicate patch. |
| P0-4 | Reply/close #4 and #3; reply to #6 with the CUDA admission behavior and the CPU-offload limitation; label #1/#2 `feature` and link this plan. | Issue state/comments reflect reality; no stale "bug" issues. |
| P0-5 | Enable Dependabot + security alerts; add a dependency review check. | Alerts page active; Dependabot PRs configured. |
| P0-6 | Re-run the release gate twice on `main` (backend + frontend + e2e) and start the legacy cutover clock. | Two consecutive green releases recorded. |

**Long-term P0 follow-up:** migrate the 42 `datetime.utcnow` sites to
`datetime.now(timezone.utc)` (or explicitly `NaiveDatetime` fields), then lift the
SQLAlchemy `<2.1` pin. Pin now, migrate deliberately.

## 3. P1 — CUDA and low-VRAM correctness (1–2 weeks)

| Step | Change | Acceptance |
|---|---|---|
| P1-1 | **CPU-offload feasibility spike (2 days).** The vendored LTX pipelines expose no `cpu_offload` hooks. Evaluate moving the transformer/VAE to CPU between stages, or conclude it is out of scope. | Written decision; if infeasible, #6 closes with the 24 GB minimum and the quantized path as the answer. |
| P1-2 | **Quantized Flux 2 (#2).** Add GGUF entries to the manifest/registry, a loader into `flux2.model.Flux2` (or a diffusers GGUF path), per-checkpoint costs, and take provenance. | Q4_K_M runs on a 12–16 GB card; quality/time compared with `backend/benchmark_models.py`; a fresh install can download it. |
| P1-3 | CUDA regression checklist: no_grad encode, VRAM admission, FP8 selection, 16/24/48 GB behavior. | Manual run on a CUDA host recorded; no OOM after hours of denoise. |
| P1-4 | CUDA cost calibration: verify the 40 GB FP8 / 18 GB Flux bases against a real card; adjust manifest estimates. | Estimates within 10% of measured peak. |

## 4. P1 — Packaging and legacy cutover (1–2 weeks)

| Step | Change | Acceptance |
|---|---|---|
| P1-5 | Bundle the backend: PyInstaller/Nuitka one-folder with Torch, or an embedded Python runtime; Tauri sidecar entry; first-run model download; macOS notarization and Windows build in CI. | A clean machine launches Studio with no Python/venv; `milimo-backend` is found by the supervisor. |
| P1-6 | Legacy cutover: after two clean packaged releases, remove the `/` fallback (or freeze it behind a plugin), delete the legacy store, update the docs/flags. | Studio is the only route; rollback flag documented; e2e updated. |
| P1-7 | Packaged-app smoke: project create/save, image + video generation (small), export, SSE progress, memory refusal, SAM optional. | Checklist green on macOS arm64 and Windows x64. |

## 5. P2 — Product gaps (4–8 weeks, parallelizable)

| Item | Work | Acceptance |
|---|---|---|
| NLE completeness | Transitions, LUTs, titles, A2 track; render + UI + tests | Export pixels match the timeline; UI round-trips |
| Embedding search | Local embedding model via the LLM plane, index, ranked results with fallback to lexical | Search finds semantically related shots; no cloud dependency |
| Shorts/lens packs | Versioned template packs, import/export, preview | Packs install and render deterministically |
| Wan 2.2 (#1) | New `WanPipeline` adapter (DiT/VAE/umT5), registry family, LoRA loader, task branch, costs, UI, baselines | A Wan checkpoint generates video and LoRAs apply; memory preflight covers it |
| M5 artifacts | Distilled/quantized LTX, MLX/FP8 lanes, quality/memory/time matrix | A chosen artifact is measurably better on the reference machine |
| LLM caveats | True socket abort for non-stream HTTP; move chained chunk-0 enhancement before the pipeline load; document Strata context; legacy popover parity | Cancellation is immediate; chunk 0 never overlaps the LTX load |
| Small polish | Per-job priority override; macOS tray icon; legacy notification parity | Settings persist; no regressions |

## 6. P3 — Quality and operations

- De-flake the 500-card perf spec (median of 3 or a tolerant threshold) and record
  the machine baseline.
- Visual baselines: add a platform matrix or document the tolerated drift.
- Delivery-lag metric (client ack) if the remaining event metrics are insufficient.
- Keep docs/AGENTS/index/log in the same commit as behavior changes.

## 7. Sequence and dependencies

```
Week 1      P0-1..P0-6  (CI green, PR/issue triage)
Weeks 2-3   P1-1..P1-4  (offload decision, quantized Flux, CUDA validation)
Weeks 4-5   P1-5..P1-7  (sidecar, packaged smoke)
Week 6      legacy cutover clock + P2 start (NLE/search/packs in parallel)
Weeks 7-12  Wan, M5 artifacts, product gaps
```

Hard dependencies:

- Legacy cutover needs two green packaged releases.
- Quantized Flux needs GGUF artifacts and a loader; it also answers #2 and most of
  #6 for 12–16 GB cards.
- Wan needs its own checkpoints; it must not be forced through LTX assumptions.
- CUDA validation needs a real 16/24/48 GB host (CI has no GPU).

## 8. Definition of done for this plan

1. `main` and `develop` are green for two consecutive releases.
2. Open issues reflect reality: #4/#3 closed, #6 answered, #1/#2 labeled and
   scheduled.
3. PR #7 closed as superseded and PR #8 merged with credit.
4. A packaged app launches on a clean machine.
5. The legacy route is removed or explicitly frozen after the cutover window.
6. Each P2 item has either shipped with tests or an explicit "won't do" decision.

## Related

- [[implementation-plan-2026-11.md]] — current status table
- [[../systems/github-issues-triage-2026-11.md]] — issue-by-issue evidence
- [[../systems/memory-management-audit-2026-11.md]] — CUDA/M5 context
- [[../systems/job-notifications-audit-2026-11.md]] — notification caveats
- [[../../docs/07_api.md]] — automation surface
