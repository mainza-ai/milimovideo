---
type: research
created: 2026-10-09
updated: 2026-10-09
tags: video-models research landscape ltx-2 wan licensing
sources:
  - research survey 2026-10-09 (web)
related:
  - ../models/ltx-2.md
  - ../pipelines/chained-generation.md
  - ../roadmap/v2-roadmap.md
---

# Research: Open-Source Video Generation Models (Oct 2026)

Survey of the open video-model landscape since LTX-2, for the v2 model-slot decision.
All post-Jan-2026 facts are web-sourced; unconfirmed items are marked.

## Landscape

- **LTX-2.5 (Lightricks, Aug 11 2026, 22B)** — direct successor to LTX-2 19B. Native
  multishot generation, keyframe pipeline, 4K HDR, joint audio-video, distilled fast
  variants (10s clip in ~6.8s on 2×GB200; community runs on 8GB VRAM). Positioned as
  an open-weights "world model" with prompt-controlled camera behavior + start/end
  keyframes. License: LTX-2.x Community (free under $10M ARR; derivatives inherit
  the license). Day-one ComfyUI + Diffusers; verified Mac MLX ports.
- **Wan 2.2 (Alibaba, Jul/Aug 2025, 27B MoE / 14B-active, 5B, 1.3B)** — the
  license-clean workhorse: **Apache 2.0**, T2V/I2V/first-last-frame, VACE/Fun-Control
  for restyle, distilled variants, FP8/GGUF/NVFP4 quants, LightX2V 4-step. Largest
  open inference ecosystem and LoRA community. 5B TI2V runs 720p at ~8GB.
- **Wan 3.0 (Alibaba, Aug 2026)** — unified T2V/I2V/first-last-frame/reference-to-video,
  30s @ 30fps, native audio, pledged Apache 2.0 — but **open weights NOT shipped as of
  Oct 9, 2026** (API-only). Watch, do not plan on.
- **MiniMax H3 / Hailuo 3.0 (Jul/Aug 2026, 33B)** — quality-leading open weights
  (#1 AA video-editing arena, #2 T2V), native stereo audio, 4–15s @ 24fps 768p,
  NVFP4 quants runnable on 12–24GB. **Hard blocker: community license excludes the
  US, EU, UK, and South Korea from local deployment** (+ $20M revenue authorization
  + attribution). Rejected for Milimo (US-based).
- **Kandinsky 6.0 Video (Oct 6 2026, Pro 29B / Lite 3B)** — joint audio-video in one
  pass (44 kHz, lip-sync), **MIT license**, ~16GB VRAM with offload. Released 3 days
  before this survey: ecosystem immature, 5s clips only, quality claims vendor-run.
  Watch for Q1 2027.
- **HunyuanVideo-1.5 (Tencent, Nov 2025, 8.3B)** — best quality-per-VRAM (13.6GB peak
  720p with offload; FP8 ~8.3GB file), T2V+I2V+keyframes+V2V editing. **Tencent
  Hunyuan Community license excludes EU/UK/SK** — a distribution liability. Rejected.
- **SkyReels-V2 (Apr 2025, 14B)** — diffusion-forcing autoregressive model; the only
  open native "infinite-length" long-video DiT besides chained workflows. Runs on
  RTX 4090-class but slow (~5–15 min per 5s clip). V3 (Jan 2026) adds reference I2V
  and talking avatar; thin sourcing.
- **StepVideo-TI2V-4B (StepFun, Jul 2026, 4B, Apache 2.0)** — dual text+image
  conditioning for consumer GPUs. Single thin source; announcement-level only.
- **CogVideoX v1.5 (Nov 2024)** and **Mochi 1 (Oct 2024, Apache 2.0)** — no 2026
  successors found; superseded for general use (Mochi retains a motion-physics niche).
- **FramePack (Apr 2025)** — a method (fixed-context next-frame prediction, 13B on
  6GB VRAM), not a model; no 2026 update.

## Comparative Assessment

| Model | Params | License | T2V/I2V/Keyframe | Long-video | Local VRAM | Verdict |
|---|---|---|---|---|---|---|
| LTX-2.5 | 22B | LTX Community (<$10M ARR) | ✅/✅/✅ | Native multishot | 8GB+ (community) | **Adopt** — direct successor |
| Wan 2.2 | 14B-act/5B/1.3B | **Apache 2.0** | ✅/✅/✅ first-last | Community chained | 8–24GB | **Adopt** — clean license, biggest ecosystem |
| Wan 3.0 | undisclosed | pledged Apache 2.0 | API-only today | 30s single pass | n/a | Watch — weights not shipped |
| MiniMax H3 | 33B | excludes US/EU/UK/KR | ✅/✅/✅ | 15s | 12–24GB | **Reject** — territory clause |
| Kandinsky 6.0 | 29B/3B | **MIT** | ✅/✅/unconfirmed | 5s clips | ~16GB offload | Watch — 3 days old |
| HunyuanVideo-1.5 | 8.3B | excludes EU/UK/SK | ✅/✅/✅ | chained | ~13.6GB | **Reject** — territory clause |
| SkyReels-V2 | 14B | open (terms unconfirmed) | ✅/✅/✅ | ✅ native autoregressive | 4090-class, slow | Niche — native long-video |

Quality standing (measured): AA video arenas (Oct 2026) are led by API/hosted models;
among downloadable weights, H3 led before its license exclusion mattered, prior leader
was LTX-2.3. Wan 2.1's old VBench #1 is stale.

## Relevance to Milimo Video

**Recommendation: complement, don't replace — LTX-2.5 22B primary, Wan 2.2 A14B
secondary.** LTX-2.5 keeps the same lineage, license, and pipeline families (TI2Vid,
keyframe, distilled) while adding native multishot (fits the storyboard engine), 4K
HDR, joint audio-video, and faster-than-real-time distilled inference — the Quantum
Alignment chained-generation work carries over. Wan 2.2 fills LTX's weak spots with
the only clean Apache 2.0 license in the quality tier, the richest LoRA/finetune
community, and mature first-last-frame + chained long-video workflows. Kandinsky 6.0
revisit Q1 2027. Explicitly rejected: MiniMax H3 (US exclusion), Wan 3.0 (not
downloadable), HunyuanVideo-1.5 (EU/UK exclusion).

## Sources

- LTX-2.5: https://github.com/Lightricks/LTX-2 · https://ltx.io/model/capabilities/world-model · https://www.marktechpost.com/2026/08/11/the-video-production-stack-now-fits-on-one-desk-ltx-2-5-launches-as-nvidia-accelerated-open-weights-world-model/
- Wan: https://github.com/ryannel/skills/blob/HEAD/skills/generative-media/wan-2-2/SKILL.md · https://blog.segmind.com/wan-3-guide-features-examples-and-what-developers-should-know/ · https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4
- MiniMax H3: https://mlq.ai/news/minimax-h3-tops-a-video-editing-arena-but-its-open-weights-exclude-us-users/ · https://runaihome.com/blog/minimax-h3-open-weights-local-ai-hardware-guide-2026/
- Kandinsky 6.0: https://paperswithcode.co/paper/2610.05608 · https://explainx.ai/blog/kandinsky-6-video-open-source-29b-audio-video-mit-license-2026
- HunyuanVideo-1.5: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- SkyReels: https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
- Leaderboards: https://rctv.com/posts/ai-video-weekly-roundup-2026-10-05/ · https://www.versely.studio/blog/state-of-ai-video-oct-2026
- Local VRAM table: https://intelligibberish.com/articles/best-local-video-generation-models-by-vram/

Unconfirmed: Step-Video-T2V exact license text; SkyReels-V2 exact terms; Wan 2.7
open-release details; Happy Horse weight availability; "daVinci-MagiHuman" (single
source, no repo/paper — treated as unverified).

## Related

- [[../models/ltx-2.md]] — the current video slot occupant
- [[../pipelines/chained-generation.md]] — Quantum Alignment, which carries over
- [[../roadmap/v2-roadmap.md]] — the v2 plan
