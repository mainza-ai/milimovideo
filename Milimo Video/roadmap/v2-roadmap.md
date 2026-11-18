---
type: roadmap
created: 2026-10-09
updated: 2026-11-17
tags: v2 roadmap planning upgrade
sources:
  - wiki/research/video-models-2026.md
  - wiki/research/image-models-identity-2026.md
  - wiki/research/world-models-3d-camera-2026.md
  - wiki/research/inference-infra-2026.md
  - wiki/research/frontend-ux-workflows-2026.md
related:
  - ../overview.md
  - ../research/video-models-2026.md
  - ../research/image-models-identity-2026.md
  - ../research/world-models-3d-camera-2026.md
  - ../research/inference-infra-2026.md
  - ../research/frontend-ux-workflows-2026.md
  - ../research/model-landscape-2026-11.md
---

# Milimo Video v2 — Design Roadmap

The plan to bring Milimo Video from its Feb 2026 state to the Oct 2026 state of the
art. Research-backed (see `wiki/research/`), license-conscious, local-first.
**No implementation** — this document is the decision record Mainza approves before
Phase 0 begins.

> ⚠️ **VERIFY flags** in this document mark claims that depend on the repo's docs
> rather than the code. The wiki already records 8 doc contradictions (see log.md).
> Phase 0 exists to resolve them before any build decision is final.

> **Sequencing update — 2026-11-16.** A code-level audit
> ([[../systems/engineering-audit-2026-11.md]]) resolved those eight
> contradictions and found that the v1 studio layer is not yet trustworthy:
> V1 order and conditioning do not round-trip, export timing/compositing/audio are
> wrong, and image/inpaint jobs bypass the GPU queue. Model swaps are therefore
> **gated** behind the fixes in
> [[implementation-plan-2026-11.md]] (Gates G2/G3). The model recommendations,
> research pages, and architecture direction below stand unchanged. Read this page
> as the destination; read the implementation plan as the road.

## Guiding Principles

0. **Local-first stays the religion.** Every 2026 survey confirmed the moat: the
   open-weights ecosystem matured *toward* local execution (GGUF video models, MLX
   DiT ports, Tauri sidecars), not away from it. Cloud stays an optional,
   explicitly-consented lane — never the hot path. (AGENTS.md law 0, unchanged.)
1. **Keep the slot architecture.** Mutual exclusion is the industry-standard pattern
   — vLLM-Omni independently re-derived it in 2026. Update the budgets (FP8 halves
   slots on NVIDIA), keep sequential CPU offload as the layer-level escape hatch.
2. **MPS-first, with one relaxation.** bfloat16 is stable on MPS since PyTorch 2.10
   — drop the mandatory float32 transformer forcing. Keep CPU VAE decode as
   default-safe. Budget ~5–8× slower than an RTX 4090 for the video slot on Mac;
   FP8/NVFP4 remain CUDA-only.
3. **License hygiene is a feature.** Territory and revenue traps eliminated three
   quality-leading models in this survey (MiniMax H3, HunyuanVideo-1.5,
   Qwen-Image-2.1). Default to Apache 2.0/MIT; anything else needs explicit review.
4. **Eval before swap.** No model changes land without the VBench-2.0 subset +
   blind A/B harness from Phase 1. Taste is not a benchmark.
5. **Complement, don't rewrite.** The storyboard engine, @trigger identity concept,
   Quantum Alignment, and SSE eventing are the durable IP. v2 upgrades the engines
   underneath them; it does not redesign what already works.

## Model Recommendations

### Video slot — LTX-2 19B → **LTX-2.5 22B** (primary) + **Wan 2.2 A14B** (complement)

- **LTX-2.5**: same lineage, same pipeline families (TI2Vid, keyframe, distilled),
  same <$10M ARR community license — no new legal burden. Adds native multishot
  generation (fits the storyboard engine shot-for-shot), 4K HDR, joint audio-video,
  prompt-controlled camera behavior + start/end keyframes, and faster-than-real-time
  distilled inference. The Quantum Alignment chained-generation work carries over;
  LTX-2.3's IC-LoRA in/outpainting is evaluable inside the same stack.
- **Wan 2.2**: the only quality-tier model with a clean **Apache 2.0** license — no
  revenue trigger, no territory exclusion. Largest open inference ecosystem
  (Diffusers, ComfyUI, LightX2V, GGUF/FP8/NVFP4, MLX), best VRAM ladder (5B TI2V at
  ~8GB; 14B at 24GB via FP8), mature first-last-frame + chained long-video
  workflows, and the richest LoRA finetuning community. Fills LTX's weak spots.
- Fallback: keep LTX-2 19B pinned while the harness validates. Watch: Kandinsky 6.0
  (MIT, joint A/V, 3 days old — revisit Q1 2027). Rejected: MiniMax H3 (US territory
  exclusion), Wan 3.0 (API-only, weights not shipped), HunyuanVideo-1.5 (EU/UK
  exclusion).

### Image slot — Flux 2 Klein 9B → **Z-Image Turbo + Qwen-Image-Edit-2511** (split)

- **Z-Image Turbo** (6.15B, Apache 2.0, 8-step DMD): arena-leading open image model,
  8–16GB VRAM, already on Apple Silicon via MLX. Becomes the fast default for
  concept-art thumbnails and pre-viz — strictly faster and lighter than Klein 9B.
- **Qwen-Image-Edit-2511** (20B, Apache 2.0): the dedicated editor. Native
  multi-reference editing replaces the IP-Adapter @trigger hack with better
  multi-person consistency; mask-based inpainting composited back untouched replaces
  the RePaint loop with a single instruction-conditioned pass.
- **Durable identity: per-character LoRA** (ai-toolkit, rank 16–32, 10–30 images,
  ~1–3h on 24GB or ~$2 cloud run). The only method that holds identity across
  poses/lighting at production quality. The @trigger-word UX stays; the mechanism
  underneath graduates from adapter hack → native multi-ref → trained LoRA.
- Fallback: FLUX.2 [dev] 32B unified (top open quality, native 10-ref) — only after
  legal review of the BFL non-commercial terms; needs 24GB+ VRAM. Rejected:
  Qwen-Image-2.1 (research-only license — hard blocker).

### Segmentation slot — **KEEP SAM 3**

- No SAM 4 exists as of Oct 2026; SAM 3 remains the SOTA foundation for exactly the
  microservice's job. Keep the port-8001 isolation (the decision record still holds).
- Add **SAM 3D Objects** (same SAM license) as a prop-previz lane: single image →
  textured mesh, sharing the SAM 3 front end.
- Spike **sam3.cpp** (2026 ggml port, license unconfirmed) as a lighter
  Apple-Silicon inference path to reduce the microservice's GPU dependency.
- Add **ProPainter** (MIT) as a dedicated video object-removal/completion lane
  alongside the generative inpaint editor. Watch Netflix VOID (Apache 2.0, best
  interaction-aware deletion, but 40GB+ VRAM — server-class).

### LLM slot — Gemma 3 → **Qwen3.x 4B–9B (rewrite) + Qwen3-VL 8B (critique)**

- **Qwen3.x 4B–9B class** for prompt rewriting (replaces/augments Gemma 3; Apache
  2.0, ungated — no HF gating friction). Keep Ollama as the runtime; keep the
  `keep_alive: 0` unload discipline. *The role plane now makes this a per-role
  choice rather than a swap; models are compared with the local eval before
  assignment — see [[../systems/llm-local-first-next-2026-11]].*
- **Qwen3-VL 8B** (6.1GB, 256K context, native video understanding) for shot analysis
  and pre-viz critique — the VLM in the VQQA refinement loop (below).

### New capability lanes (Phase 3)

- **3D previz**: TRELLIS.2 (MIT, image→PBR mesh) for sets/props; TripoSG (MIT,
  8–12GB) as low-VRAM fallback. Hunyuan3D-2.1 only with EU/UK/KR license caveat.
- **Geometry**: Depth Anything 3 Small/Base (Apache-2.0) — any-view depth+pose+metric
  depth in one call, for depth-aware effects and camera planning.
- **Camera**: comes free with LTX-2.5 (prompt + keyframes); precise-trajectory
  adapters (AC3D-style) stay on the watch list.

## Architecture Deltas

| Area | v1 (Feb 2026) | v2 delta |
|---|---|---|
| Video backend | Custom `ModelManager` + diffusers | **Keep**, add FP8 variant paths on NVIDIA (official exports load in plain diffusers); **evaluate SGLang-diffusion as an optional NVIDIA backend** (native LTX-2.x, MPS platform listed, ModelOpt quant); keep diffusers on MPS |
| Apple Silicon video | PyTorch MPS + workarounds | **Spike an MLX backend** for the video slot (community int4/int8 DiT ports) — the only 2026 route to materially faster/bigger models on Mac |
| MemoryManager | Mutual-exclusion slots (76/36/49GB) | **Keep design**; refresh budgets with FP8 numbers (video ~40–45GB FP8 on CUDA); add sequential CPU offload as documented escape hatch |
| MPS law | float32 forcing + CPU VAE decode | **Relax**: bf16 now stable (drop float32 forcing on PyTorch ≥2.10); keep CPU VAE decode default-safe; FP8 stays NVIDIA-only |
| Model distribution | Manual safetensors downloads | Ship **GGUF + compressed-tensors** artifacts (community + HF-native standards); keep safetensors bf16 |
| SAM service | Port 8002, PyTorch | Keep; spike sam3.cpp path; add SAM 3D Objects lane |
| Frontend | React 18, DOM timeline, web-only | **React 19.2+ + React Compiler**; canvas timeline layer + virtualization; **Tauri 2 sidecar** packaging (Electron escape hatch behind `shell-api`) |
| State | Zustand 7-slice | **Keep**; isolate playhead/transient state into a dedicated hot slice with strict selectors |
| Agents | None | **Supervisor crew on LangGraph** (Python): Director → Cinematographer → Continuity (CANVAS-like reference store) → Reviewer (VQQA VLM loop); **MCP server** over timeline/storyboard; HITL interrupts before generation spend |
| Eval | None | **VBench-2.0 subset + FFmpeg sanity + blind A/B UI** (AISBench mini-subset path) — gates every model swap |

## UI/UX Direction

The full design is now in [[frontend-refactor-2026-11.md]] and
[[../systems/studio-shell.md]], backed by
[[../research/frontend-video-platforms-2026.md]].

- **One project graph, three lenses.** Story (script + board + shot list),
  Generate (shot workspace + takes + ingredients), Edit (stage + timeline). Board
  position never defines order; `sequence_index` does. A shot card, shot-list row,
  and timeline clip are the same object.
- **Board + timeline are peers.** The board is the place to plan meaning,
  continuity, references, and variants; the timeline is the place to finish time,
  audio, and captions. Neither is a modal.
- **Master References + immutable takes.** Promote Elements to first-class
  character/location/object/style/voice references; every generation becomes a
  take; the shot selects one; review compares them.
- **Copilot/command dock bound to selection and time range.** The agent shows
  context chips, a plan with model/objects/time/VRAM, a contact-sheet/timeline
  diff, and requires approval by default. Applied changes are undoable patches.
- **Transcript-first rough-cut mode** for dialogue and shorts (Descript pattern),
  feeding the existing script→scenes→shots engine.
- **Review is inside the studio**: version stacks, compare viewer, frame-accurate
  comments, approvals, activity, share links, and an `@Agent` action that proposes
  rather than silently edits.
- **Shorts workspace** as a project preset: source/transcript → candidate moments
  → reframe 9:16/1:1/4:5 → captions/brand → variant matrix → deliver.
- **Canvas craft**: LOD, viewport culling, minimap, areas, typed links, pins,
  list/table equivalent, command palette, focus mode. Kubona (Apache 2.0) is the
  interaction reference; do not import its quantum domain.
- **Progressive lenses**: Story Board first; Pipeline (node recipe), Blocking
  (top-down camera diagram), and Previz 3D are later lenses over the same graph.
- **Performance**: 500 cards / 200 clips at 60fps; canvas layer for dense timeline
  media, DOM for handles/labels; isolated playhead; React Compiler; route-level
  code splitting.
- **MCP server** exposing board/timeline ops — external agents can drive the
  studio with the same plan/approval contract. Product boundary: Milimo Video
  stays standalone; Kubona is a separate research platform.

## New Workflows Enabled

1. **Agentic direction**: Director agent turns a script into a shot plan;
   Cinematographer specs each shot; Continuity agent checks every generation against
   the CANVAS-like character/location store; Reviewer runs the VQQA VLM critique →
   prompt refinement loop. Human approves at each spend point.
2. **Character LoRA pipeline**: define a Character element → collect/generate 10–30
   refs → one-click LoRA training (local 24GB or ~$2 cloud lane) → `@Name` triggers
   the trained identity everywhere, video and stills.
3. **3D previz**: concept art → TRELLIS.2 mesh → block the scene in 3D before
   spending video generation budget.
4. **Object removal lane**: ProPainter for clean plate work without generative
   reinterpretation; VOID later if VRAM allows.
5. **Eval-gated upgrades**: every model swap ships with its VBench-2.0 subset
   scores and blind A/B results in the wiki — the decision trail compounds.

## Phased Plan

- **Phase 0 — Verify against code (1–2 weeks).** **DONE 2026-11-16** — the audit's
  "Phase 0 Verification" table records the code truth for all eight contradictions
  (cfg_scale, upscale, endpoint names, router count, `/track/remove`,
  model_engine↔memory_manager, Flux variants, inpaint SSE/Asset detail). New
  finding: the v1 editor is not a trustworthy base yet. **Phase 0.5** is therefore
  the stabilization program in [[implementation-plan-2026-11.md]] (security,
  timeline truth, export fidelity, unified GPU queue) — Gates G0–G3.
- **Phase 1 — Eval harness + model swaps (4–6 weeks).** **Gated by G2/G3.** Stand up VBench-2.0 subset
  + FFmpeg sanity + blind A/B UI first. Then, each gated by the harness: (a) video
  slot → LTX-2.5 22B (FP8 path on NVIDIA); (b) image slot → Z-Image Turbo (fast)
  + Qwen-Image-Edit-2511 (editor); (c) LLM slot → Qwen3.x + Qwen3-VL 8B. Wan 2.2
  lands as the second video engine behind a feature flag.
- **Phase 2 — Architecture (4–6 weeks).** Slot budget refresh + sequential-offload
documented; SGLang-diffusion spike (NVIDIA); MLX video-backend spike (Mac);
  GGUF/compressed-tensors distribution artifacts; sam3.cpp spike; ProPainter lane.
- **Phase 3 — UX & agentic workflows (6–8 weeks, now 12–14 weeks for the full
  shell).** Run the F0–F4 program in [[frontend-refactor-2026-11.md]]: studio shell,
  Story/Board, Edit/Timeline, Generate/Takes, Agent/Review/Library/Deliver, then
  Shorts and the Pipeline/Blocking/3D lenses. React 19 + Compiler;
  canvas timeline + virtualization; copilot with plan/diff/approval;
  transcript-first mode; LangGraph supervisor crew + MCP server; Tauri 2 sidecar.
- **Stretch (post-v3 or as spikes):** TRELLIS.2 previz lane, Depth Anything 3,
  SAM 3D Objects, Kandinsky 6.0 revisit (Q1 2027), Wan 3.0 if weights ship.

## Risks

- **License traps** — the single biggest 2026 hazard. Three quality-leading models
  were rejected on licensing alone. Every adoption needs the license re-verified
  against the actual weight card, not a roundup.
- **Apple Silicon perf gap** — ~5–8× slower than an RTX 4090 for the video slot;
  MLX is community-grade. MPS-first remains a correctness story, not a speed story.
- **Ecosystem immaturity** — Kandinsky 6.0 (3 days old), Zing-0.5/ABot-World-0
  (licenses unconfirmed), sam3.cpp (unconfirmed). Adopt nothing without a
  second source.
- **Scope creep** — the v1 lesson. The phased plan is the guardrail: no Phase 3
  work starts before Phase 1's harness exists.
- **Doc/code divergence** — 8 known contradictions; Phase 0 is non-optional.
- **NVIDIA/Hugging Face acquisition** (Sep 2026) — packaging politics to watch; no
  format disruption observed, but GGUF independence is a hedge.

## Open Questions for Mainza

1. **Target hardware**: what is the minimum machine v2 must run well on? (This sets
   every VRAM budget — e.g., 16GB VRAM NVIDIA vs 32GB Mac vs 128GB Mac.)
2. **MPS-first or CUDA-primary?** Keep Apple Silicon as the first-class citizen, or
   optimize for NVIDIA and treat Mac as compatible?
3. **Cloud fallback stance**: the mv_plan proposal (Veo/Imagen/Gemini fallbacks) is
   still marked not-implemented. Does v2 keep local-only, or does an explicit
   opt-in cloud lane get built?
4. **Revenue vs license traps**: the LTX community license has a $10M ARR trigger.
   Is that acceptable long-term, or should Wan 2.2 (Apache 2.0) become the primary
   video engine instead?
5. **LoRA training UX**: local-only training (24GB+ cards) or a first-party cloud
   training lane (~$2/run) for character LoRAs?
6. **Agentic scope**: should the supervisor crew live inside Milimo Video's backend,
   or run as a separate local service driving the studio over MCP? (Product boundary:
   Milimo Video is standalone — no coupling to Milimo Quantum/Kubona, which is a
   separate scientific quantum research/discovery platform.)
7. **Packaging**: Tauri 2 sidecar now, or stay browser-based for v2 and package later?

## Related

- [[implementation-plan-2026-11.md]] — the sequenced stabilization + delivery plan
- [[frontend-refactor-2026-11.md]] — the studio shell redesign program
- [[../systems/studio-shell.md]] — target front-end architecture
- [[../research/frontend-video-platforms-2026.md]] — platform/canvas research
- [[../systems/engineering-audit-2026-11.md]] — the code-level evidence
- [[../research/video-models-2026.md]]
- [[../research/image-models-identity-2026.md]]
- [[../research/world-models-3d-camera-2026.md]]
- [[../research/inference-infra-2026.md]]
- [[../research/frontend-ux-workflows-2026.md]]
- [[../overview.md]] — what v1 is
