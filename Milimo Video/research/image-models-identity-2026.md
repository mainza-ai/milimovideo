---
type: research
created: 2026-10-09
updated: 2026-10-09
tags: image-models research landscape identity-consistency licensing
sources:
  - research survey 2026-10-09 (web)
related:
  - ../models/flux-2.md
  - ../concepts/story-elements.md
  - ../pipelines/inpainting.md
  - ../roadmap/v2-roadmap.md
---

# Research: Open-Source Image Models + Identity Consistency (Oct 2026)

Survey of the open image-model landscape and character/object consistency tech, for
the v2 image-slot decision. All post-Jan-2026 facts are web-sourced; unconfirmed
items are marked.

## Landscape

- **Z-Image Turbo (Alibaba Tongyi-MAI, Nov 2025, 6.15B, Apache 2.0)** — 8-step
  DMD-distilled; ranked #1 open-weights on the Artificial Analysis text-to-image
  arena (ahead of FLUX.2 [dev], HunyuanImage 3.0, Qwen-Image). FP8/GGUF on 8–16GB
  VRAM; runs on Apple Silicon via MLX (~14s on Mac). The 2026 efficiency leader.
- **FLUX.2 [dev] (Black Forest Labs, Nov 2025, 32B) + FLUX.2 [klein] (Jan 2026,
  4B/9B)** — dev is the open quality ceiling and unifies generation + instruction
  editing + **native multi-reference conditioning (up to 10 refs) in one model** —
  the architectural successor to the IP-Adapter @trigger pattern. License: BFL
  non-commercial for [dev] (paid above revenue thresholds); klein commercial-allowed
  below thresholds. Needs ~24GB ([dev]) / ~13GB / ~8GB (klein 9B/4B).
- **Qwen-Image-2512 (Alibaba, Jan 2026, ~20B, Apache 2.0)** — top open model on blind
  human-preference rounds per 2026 roundups; photorealism + in-image text (EN/CN).
  ~24GB VRAM; GGUF Q4 ~12–13GB.
- **Qwen-Image-2.1 (Alibaba, Sep 2026, 7B)** — unified generation+editing, native
  transparency, 10 reference images, 2K output — but **Qwen Research License:
  research/evaluation only, no commercial use**. Hard blocker. Rejected.
- **Qwen-Image-Edit-2511 (Dec 2025, 20B, Apache 2.0)** — the license-clean editor:
  mask-based inpainting, instruction editing, multi-image editing with improved
  character/multi-person consistency. FP8 ~16GB, GGUF Q4 ~14GB. Community consensus:
  beats FLUX.1 Kontext on free-form instruction following.
- **GLM-Image (Z.ai, Jan 2026, 9B AR + 7B diffusion, MIT/Apache-2.0)** — hybrid with
  built-in identity consistency, multi-subject consistency, best-in-class typography.
  ~20GB VRAM.
- **HunyuanImage 3.0-Instruct (Tencent, Jan 2026, 80B MoE / 13B active, Apache
  2.0)** — largest open image model, but 40GB+ VRAM and non-commercial third-party
  ComfyUI integration. Not practical for consumer GPUs.
- **Stable Diffusion 4: no release found as of Oct 2026** (current is SD 3.5, Oct
  2024). Stability AI's 2026 went to music. SDXL/SD3.5 remain the LoRA ecosystem kings.

## Identity Consistency Tech

- **Native multi-reference conditioning is the 2026 best practice** (FLUX.2 [dev] 10
  refs; Qwen-Image-Edit multi-image): training-free, no adapter plumbing — directly
  replaces the IP-Adapter @trigger workflow.
- **IP-Adapter lineage** (Plus/FaceID) remains the workhorse on Flux.1/SDXL, but
  Flux 2 absorbed the capability natively; no adapter needed.
- **LoRA remains the practical default for durable identity**: rank 16–32, 10–30
  captioned images, rare trigger token, ~40MB files. ai-toolkit is the default trainer
  (Flux.1/Flux.2/Klein); Flux.1 LoRA trains in 1–3h on 24GB; Klein 9B needs 32–48GB
  (cloud lane ~$2/run). Only method that holds identity across poses/lighting at
  production quality.
- **PuLID/FaceID/InstantID** are face-specific; PuLID drags a non-commercial
  InsightFace license — flag for review. Less relevant to Milimo's object/character use.
- **ACE++** (Jan 2026) is legacy — the ecosystem moved to native unified edit models
  (UniGen-1.5, OmniGen2 lineage).

## Comparative Assessment

| Model | Params | License | Identity mechanism | Inpainting | Local VRAM | Verdict |
|---|---|---|---|---|---|---|
| Z-Image Turbo | 6.15B | **Apache 2.0** | refs (community) | via edit models | ~8–16GB | **Adopt** — fast default |
| Qwen-Image-Edit-2511 | 20B | **Apache 2.0** | native multi-ref | ✅ mask + instruction | ~14–16GB | **Adopt** — editor slot |
| FLUX.2 [dev] | 32B | BFL non-commercial | native 10-ref | ✅ native | ~24GB+ | Fallback — license review |
| FLUX.2 [klein] 9B/4B | 9B/4B | BFL (commercial < thresholds) | native multi-ref | ✅ | ~13GB/~8GB | Alt fallback |
| Qwen-Image-2512 | 20B | **Apache 2.0** | strong base | — | ~24GB (Q4 ~13GB) | Quality alt |
| Qwen-Image-2.1 | 7B | research-only | native 10-ref | ✅ | low | **Reject** — license |

## Relevance to Milimo Video

**Recommendation: restructure the image engine — Z-Image Turbo (Apache 2.0) as the
fast default for concept art/thumbnails, Qwen-Image-Edit-2511 (Apache 2.0) as the
dedicated inpaint/editor replacing the RePaint loop, and per-character LoRA
(ai-toolkit, rank 16–32) as the durable identity mechanism.** Z-Image Turbo is
arena-leading, 8 steps, sub-second-class, 8–16GB, already on Apple Silicon via MLX —
a strict upgrade for concept art. Qwen-Image-Edit-2511's native multi-reference
editing does what the IP-Adapter @trigger hack does today but better (multi-person
consistency, mask inpainting composited back untouched). LoRA per character (10–30
images, ~1–3h on 24GB or ~$2 cloud) is the only production-grade long-lived identity.
All Apache 2.0/MIT — Milimo's open-source distribution stays clean. Fallback: FLUX.2
[dev] 32B unified (needs legal review + 24GB+ VRAM).

## Sources

- Arenas/roundups: https://localaimaster.com/blog/best-local-image-models-compared · https://medium.com/@moksh45/self-hosting-ai-image-generators-in-2026-flux-hunyuanimage-qwen-and-more-413e7ee6efc4
- FLUX.2: https://github.com/youssoufall2002-hash/flux2 · https://huggingface.co/unsloth/FLUX.2-dev-GGUF
- Z-Image Turbo: https://tech-insider.org/z-image-turbo-local-setup-2026/
- Qwen-Image-2512: https://venturebeat.com/technology/open-source-qwen-image-2512-launches-to-compete-with-googles-nano-banana-pro
- Qwen-Image-2.1 license: https://pexo.ai/blog/qwen-image-2-1-alternatives-6421 · https://dev.to/techaiwire/qwen-image-21-ships-7b-weights-under-a-research-license-1iag
- Qwen-Image-Edit: https://www.thundercompute.com/blog/qwen-image-edit-comfyui
- LoRA training: https://tech-insider.org/train-custom-ai-image-lora-2026/ · https://github.com/waltm/lora-dataset-studio
- Apple Silicon: https://github.com/newideas99/Ultra-Fast-Image-Generation-Mac-Silicon-Z-Image
- GLM-Image: https://venturebeat.com/technology/z-ais-open-source-glm-image-beats-googles-nano-banana-pro-at-complex-text

Unconfirmed: UniPic-3 license; Z-Image-Edit variant status; FLUX.2 [klein] commercial
threshold terms (verify in BFL license text); Qwen-Image-Edit VRAM figures are
community-reported; arena rankings are vendor/community-reported.

## Related

- [[../models/flux-2.md]] — the current image slot occupant
- [[../concepts/story-elements.md]] — the @trigger system this replaces/upgrades
- [[../pipelines/inpainting.md]] — the RePaint loop this replaces
- [[../roadmap/v2-roadmap.md]] — the v2 plan
