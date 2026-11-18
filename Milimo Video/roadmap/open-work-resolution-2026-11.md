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
| CI on `main`/`develop` | **GREEN** | Fixed by pinning the DB/vendor/FastAPI stack; `6a36cc8` is green on both branches. The failure was reproducible with SQLModel 0.0.50 + SQLAlchemy 2.1.9 (`Datetime values must have timezone information`); 181 tests pass locally. |
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

## 2. Market scan (2026-11-17)

The full model/workflow matrix and sources are in
[[../research/model-landscape-2026-11]]. The decisions that change this plan:

- **LTX-2.5 (Aug 2026) is the strategic 32 GB+ video upgrade.** 22B, native
  synchronized audio/multishot. The official diffusers `LTX2Pipeline` has
  `enable_model_cpu_offload()` and `vae.enable_tiling()` — the supported offload
  path, instead of bolting it onto the vendored pipeline. Official minimum is
  32 GB VRAM; 8–16 GB int8 reports are community experiments. Support is not in a
  diffusers release yet; install from main or `LTX-2.5-Diffusers`.
- **Wan 2.2 TI2V-5B is the permissive low-VRAM engine.** Apache 2.0, diffusers,
  native 720p/24 fps text+image-to-video; ~8 GB with ComfyUI offload. Wan 2.5+ is
  closed, so 2.2 is the ceiling for open weights.
- **Do not bundle license-restricted models.** HunyuanVideo 1.5 excludes
  EU/UK/KR; MiniMax H3 excludes US/EU/UK/KR; Qwen-Image 2.1 is research-only.
  Support them only as optional, user-installed engines.
- **Interoperability is the workflow.** ComfyUI is the graph/experimentation
  surface; OTIO/FCPXML/Resolve XML is the finishing handoff. Build those before
  trying to clone every NLE feature.
- **Audio/upscale lanes are commodity.** Qwen3-TTS (Apache 2.0) and SeedVR2
  (Apache 2.0) should be optional stages, not bespoke models.

## 3. P0 — Restore a trustworthy baseline (1–2 days)

**Goal:** green CI on `main` and `develop`, stale contributions resolved, GitHub
issues triaged. Everything else is blocked on this.

| Step | Change | Acceptance |
|---|---|---|
| P0-1 | Pin the known-good DB stack in `backend/requirements.txt`: `sqlmodel==0.0.31`, `sqlalchemy>=2.0.14,<2.1`, plus the flux2/FastAPI/Pydantic versions; add `pip check`. | **Done** — CI backend green on `main` and `develop` at `6a36cc8`; `pip check` clean. |
| P0-2 | Apply PR #8 (`torch.no_grad()` around generator consumption, `.detach()` before CPU) and add a regression test. | **Done** — merge `28729c5` + `tests/test_encode_video.py`; CPU test passes. CUDA manual check remains. |
| P0-3 | Close PR #7 as superseded with line references. | **Done** — closed with the `tasks/image.py:98` / `tasks/video.py:290` evidence. |
| P0-4 | Reply/close #4 and #3; reply to #6 with the CUDA admission behavior and the CPU-offload limitation; label #1/#2. | **Done** — #4/#3 closed; #6/#2/#1 replied and labeled `enhancement`/`help wanted`. |
| P0-5 | Enable Dependabot + security alerts. | **Done** — `.github/dependabot.yml`; vulnerability alerts enabled (204), Dependabot alerts 200. |
| P0-6 | Re-run the release gate twice on `main` and start the legacy cutover clock. | **CI gate met** — two consecutive green CI cycles on both branches (`6a36cc8`, `48a5d75`). No release tags exist, so the formal two-release cutover still needs tagged releases. |

**Long-term P0 follow-up:** migrate the 42 `datetime.utcnow` sites to
`datetime.now(timezone.utc)` (or explicitly `NaiveDatetime` fields), then lift the
SQLAlchemy `<2.1` pin. Pin now, migrate deliberately.

## 4. P1 — Video engines, CUDA, and shipping (3–5 weeks)

| Step | Change | Acceptance |
|---|---|---|
| P1-1 | **Adopt the diffusers LTX-2.5 engine.** Spike `LTX-2.5-Diffusers` / diffusers `main` with `LTX2Pipeline`, `enable_model_cpu_offload()`, `vae.enable_tiling()`, and int8/W4A8 quants; map it behind the current task interface, keep the vendored LTX-2 engine as a fallback. | 32 GB CUDA runs bf16/FP8 with offload/tiling; 16 GB int8 is validated as an explicit experimental target, not promised; Mac falls back cleanly. This is the supported resolution for #6. |
| P1-2 | **Add Wan 2.2 TI2V-5B as the low-VRAM CUDA engine.** Diffusers adapter, Apache 2.0 license gate, registry/manifest, costs, UI, 720p/24 baseline. | 8–24 GB CUDA generates 720p; preflight covers it; it does not disturb LTX memory accounting. |
| P1-3 | **Quantized Flux 2 (#2).** Upgrade diffusers to a Klein-capable release (0.37.1+), add GGUF manifest/registry entries, `Flux2KleinPipeline` + `GGUFQuantizationConfig`, and a quantized/offloaded Qwen3 text encoder. | Q4_K_M image generation on 12–16 GB; quality/time compared with `backend/benchmark_models.py`. |
| P1-4 | CUDA regression checklist: no_grad encode, VRAM admission, FP8/quant selection, 8/16/24/48 GB behavior; calibrate the manifest estimates. | Manual run recorded on a CUDA host; estimates within 10%. |
| P1-5 | Bundle the backend: PyInstaller/Nuitka one-folder with Torch or an embedded runtime; Tauri sidecar; first-run model download; macOS notarization and Windows build in CI. | A clean machine launches Studio with no Python/venv. |
| P1-6 | Legacy cutover: after two clean packaged releases, remove `/` or freeze it behind a plugin; update docs/flags/e2e. | Studio is the only route; rollback documented. |
| P1-7 | Packaged-app smoke: create/save, small image + video generation, export, SSE, memory refusal, SAM optional. | Checklist green on macOS arm64 and Windows x64. |

**P1-1 replaces the old “CPU offload spike.”** Bolting offload onto the vendored
LTX-2 pipeline is not viable; the diffusers LTX-2.5 path already implements it and
is the supported upgrade.

## 5. P2 — Workflow and product gaps (6–10 weeks, parallelizable)

| Item | Work | Acceptance |
|---|---|---|
| **OTIO/XML finishing handoff** | OTIO export first, then FCPXML/Resolve XML; import for round-trips | A Resolve/Premiere timeline opens with correct cuts, trims, and audio; round-trip preserves V1/V2/A1 |
| **ComfyUI bridge** | Import/export a ComfyUI workflow, submit it through the queue, ingest outputs and provenance | An LTX-2.5/Wan workflow runs from Studio and its output becomes a take |
| NLE completeness | Transitions, LUTs, titles, A2 track (after the OTIO handoff) | Export pixels match the timeline; UI round-trips |
| Embedding search | Local embedding model via the LLM plane, index, ranked results with lexical fallback | Semantic shot search works offline |
| Shorts/lens packs | Versioned template packs, import/export, preview | Packs install and render deterministically |
| Wan 2.2 family | S2V speech-to-video (with CosyVoice/Qwen3-TTS), Animate character replacement, LoRA loading | Spoken dialogue/lip-sync and character replacement work as shots |
| Qwen3-TTS voice lane | Apache 2.0 TTS role: voice design/clone, dialogue and voiceover takes, aligned captions | A script line becomes a selectable audio take without cloud |
| SeedVR2 upscale | One-step 3B upscaler as an optional post stage (16–24 GB; BlockSwap) | 2K/4K output without re-running the video model |
| Image lane upgrade | Z-Image Turbo (6B, Apache 2.0, 8-step, MLX) for thumbnails/pre-viz; Qwen-Image-Edit-2511 (20B, Apache 2.0) for multi-reference editing | Fast drafts on 8–16 GB; reference edits replace the IP-Adapter trigger workaround |
| Optional external engines | License-gated HunyuanVideo 1.5, MiniMax H3, Qwen-Image 2.1; user-supplied only | Clear license acknowledgement; never bundled or default |
| M5 artifacts | Distilled/quantized LTX, MLX/FP8 lanes, quality/memory/time matrix | A chosen artifact is measurably better on the reference machine |
| LLM caveats | Socket abort for non-stream HTTP; pre-load chunk-0 enhancement; Strata context; legacy popover parity | Cancellation is immediate; chunk 0 never overlaps the LTX load |
| Small polish | Per-job priority override; macOS tray icon; legacy notification parity | Settings persist; no regressions |

## 6. P3 — Quality and operations

- De-flake the 500-card perf spec (median of 3 or a tolerant threshold) and record
  the machine baseline.
- Visual baselines: add a platform matrix or document the tolerated drift.
- Delivery-lag metric (client ack) if the remaining event metrics are insufficient.
- Keep docs/AGENTS/index/log in the same commit as behavior changes.

## 7. Sequence and dependencies

```
Week 1       P0-1..P0-6  (CI green, PR/issue triage)
Weeks 2-4    P1-1       (LTX-2.5 diffusers/offload spike + migration)
Weeks 2-3    P1-2       (Wan 2.2 5B engine, in parallel)
Weeks 4-5    P1-3/P1-4  (quantized Flux, CUDA validation)
Weeks 5-7    P1-5..P1-7  (sidecar, packaged smoke, cutover clock)
Weeks 8-14   P2         (OTIO/ComfyUI first, then NLE/audio/upscale/Wan family)
```

Hard dependencies:

- Legacy cutover needs two green packaged releases.
- LTX-2.5 needs diffusers `main`/the `LTX-2.5-Diffusers` repo until a release ships;
  this is a dependency upgrade, not just a model download.
- Wan 2.2 5B is Apache 2.0 and diffusers-supported; its 14B/S2V/Animate variants
  come later. Wan 2.5+ has no open weights.
- Quantized Flux needs diffusers 0.37.1+, the 5.7 GB GGUF artifact, and a
  quantized/offloaded Qwen3 encoder.
- License-gated engines (Hunyuan 1.5, MiniMax H3, Qwen-Image 2.1) must not be
  bundled; they need a user-installed plugin path and a license gate.
- CUDA validation needs a real 8/16/24/48 GB host (CI has no GPU).

## 8. Definition of done for this plan

1. `main` and `develop` are green for two consecutive releases.
2. Open issues reflect reality: #4/#3 closed, #6 answered, #1/#2 labeled and
   scheduled.
3. PR #7 closed as superseded and PR #8 merged with credit.
4. A packaged app launches on a clean machine.
5. The legacy route is removed or explicitly frozen after the cutover window.
6. Each P2 item has either shipped with tests or an explicit "won't do" decision.

## Related

- [[../research/model-landscape-2026-11]] — latest models, licenses, hardware, workflows
- [[implementation-plan-2026-11.md]] — current status table
- [[../systems/github-issues-triage-2026-11.md]] — issue-by-issue evidence
- [[../systems/memory-management-audit-2026-11.md]] — CUDA/M5 context
- [[../systems/job-notifications-audit-2026-11.md]] — notification caveats
- [[../../docs/07_api.md]] — automation surface
