---
type: research
created: 2026-11-17
updated: 2026-11-17
tags: model-landscape ltx-2.5 wan2.2 hunyuan minimax qwen image tts upscale comfyui otio
sources:
  - https://github.com/Lightricks/LTX-Video
  - https://huggingface.co/Lightricks/LTX-2.5
  - https://huggingface.co/Lightricks/LTX-2.5-Diffusers
  - https://github.com/Wan-Video/Wan2.2
  - https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
  - https://huggingface.co/MiniMaxAI/MiniMax-H3
  - https://huggingface.co/Qwen/Qwen-Image-2.1
  - https://github.com/QwenLM/Qwen3-TTS
  - https://github.com/comfyanonymous/ComfyUI
  - https://opentimelineio.readthedocs.io/
related:
  - ../roadmap/open-work-resolution-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
  - ../systems/memory-management-audit-2026-11.md
  - ../systems/github-issues-triage-2026-11.md
---

# Model and Workflow Landscape (2026-11-17)

A scan of the current open-weight video/image/audio models and the workflows the
AI-film community actually uses, to re-baseline the roadmap. The guiding rules are
local-first, permissive licensing, and a path that runs on 8–32 GB CUDA as well as
Apple Silicon.

## 1. Video models

| Model | Released | Params | Native audio | License | Practical hardware | Decision |
|---|---|---|---|---|---|---|
| **LTX-2.5** (Lightricks) | Aug 11, 2026 | 22B DiT + text encoder | **Yes**, synchronized, multishot | LTX-2.x Community: free commercial under $10M revenue; gated weights | **Official minimum 32 GB VRAM** (FP8); int8/W4A8 + offload community reports 8–16 GB, experimental; 35.9–71.4 GB files | **Primary upgrade target for 32 GB+.** Official diffusers path with `enable_model_cpu_offload()` and `vae.enable_tiling()` |
| **LTX-2 / 2.3** (current) | Jan–2026 | 19B + Gemma 3 | Yes | LTX-2.x Community | 76 GB unified on MPS; FP8 40 GB CUDA | Keep as the current engine/fallback until 2.5 is integrated |
| **Wan 2.2** (Alibaba) | Jul 28, 2025 | A14B MoE (27B total/14B active); TI2V-5B | No (S2V variant is audio-driven) | **Apache 2.0** | A14B 80 GB single-GPU official, 24 GB at 640² fp8; TI2V-5B 24 GB official, ~8 GB with ComfyUI offload | **Add as the low-VRAM CUDA engine.** Diffusers + ComfyUI support; S2V/Animate variants |
| **HunyuanVideo 1.5** (Tencent) | Nov 2025 | 8.3B | No | Tencent Community: **excludes EU, UK, South Korea** | 14 GB with offload; 24 GB comfortable | Do **not bundle**; optional user-provided engine only |
| **MiniMax H3** | Aug 3, 2026 | 33B omni (T2V/I2V/audio/video) | Yes | MiniMax H3 Community: **excludes US, EU, UK, South Korea** | 8 GB via FreeVideo/VDN offload, high-end otherwise | Do **not bundle**; optional external engine only |
| **Wan 2.5 / 2.6 / 3.0** | 2026 | — | Yes | **No public weights** | API/cloud only | Out of scope for a local-first app |

### What this means

- **LTX-2.5 is the strategic video upgrade for 32 GB+**, not a new feature flag. The
  official `LTX2Pipeline`/`LTX2ImageToVideoPipeline` in diffusers supports
  `enable_model_cpu_offload()` and `pipe.vae.enable_tiling()`, which is the
  supported CPU-offload path issue #6 asked for. It also unlocks int8/W4A8
  community quants. LTX-2.5 support is not in a diffusers release yet; it installs
  from `main` or the `Lightricks/LTX-2.5-Diffusers` repo. The 8–16 GB int8 reports
  are community experiments, not an official floor.
- **Wan 2.2 TI2V-5B is the permissive low-VRAM answer.** Apache 2.0, diffusers and
  ComfyUI support, 720p/24 fps, text+image-to-video in one model. It is a better
  fit for 8–24 GB CUDA than forcing a 19–22B LTX model through offload.
- **License-gated models are not defaults.** HunyuanVideo 1.5 and MiniMax H3 are
  excluded in major markets; Qwen-Image 2.1 is under a research-only license.
  Supporting them as optional, user-installed engines avoids licensing the app
  itself into a corner.
- **Wan 2.5+ is closed.** Do not plan an open-weight integration.

## 2. Image models

| Model | Params | License | Notes | Decision |
|---|---|---|---|---|
| **Flux 2 Klein 9B** (current) | 9B | FLUX non-commercial | Custom pipeline, IP-Adapter, RePaint, AE refs | Keep as the primary image engine |
| **FLUX.2 GGUF** (unsloth/leejet) | 9B, Q2–Q8 | FLUX non-commercial | 5.7 GB Q4_K_M; diffusers single-file GGUF | Implement for low-VRAM image generation |
| **Z-Image Turbo** (Tongyi-MAI) | 6B, 8-step DMD | **Apache 2.0** | 16 GB FP16, ~8 GB quantized; MLX variant; strong bilingual text | **Fast image lane candidate** for thumbnails/pre-viz and eventually the default |
| **Qwen-Image-Edit-2511** | 20B | **Apache 2.0** | 24 GB with FP8, GGUF ~17 GB; multi-reference editing | **Editing lane candidate**, replaces the IP-Adapter trigger workaround |
| **Qwen-Image 2.1** | 7B visual | **Qwen Research License (non-commercial)** | Unified generation + editing, NVFP4 | Do **not** adopt; optional user engine only |

The image lane is not the current bottleneck. Flux 2 + GGUF covers the 6–12 GB
case once the quantized path exists. The next image upgrade is a split:
**Z-Image Turbo** for fast drafts/thumbnails (Apache 2.0, 6B, 8 steps) and
**Qwen-Image-Edit-2511** for multi-reference editing (Apache 2.0, 20B, FP8/GGUF).
Qwen-Image 2.1 is research-only and is not adopted.

## 3. Audio, segmentation, upscaling

| Model | Role | License | Notes |
|---|---|---|---|
| **LTX-2.5 built-in audio** | Dialog/SFX/ambience synced to video | LTX community | Already part of the video model |
| **Qwen3-TTS** | Voice design, 3 s voice clone, 10 languages | **Apache 2.0** | 0.6B–1.7B; strong candidate for the dialogue/voiceover lane |
| **Wan 2.2 S2V-14B** | Speech-to-video / lip-sync | Apache 2.0 | 80 GB single-GPU official; pairs with CosyVoice |
| **SAM 3** | Promptable concept segmentation + tracking | Meta custom license | Already integrated (850M params) |
| **SeedVR2 3B** | One-step video restoration/upscale to 2K/4K | **Apache 2.0** | Official ComfyUI nodes; 16–24 GB, BlockSwap on CPU | 

## 4. Workflows

The community pattern is consistent across tools:

1. **Script → shot plan** (LLM), often with character/location sheets.
2. **Storyboard/animatic** with shot cards and takes.
3. **Per-shot generation** through a queue (ComfyUI is the de facto graph engine;
   LTX-2.5, Wan 2.2, upscalers, and LoRAs all have native nodes).
4. **Review and select takes**, then **export an editable timeline** to a finishing
   NLE via **OTIO/FCPXML/Resolve XML**.
5. **Finish** in Resolve/Premiere/FCP (color, titles, sound), not in the generator.

Implications for Milimo:

- **Interoperability beats reimplementation.** An OTIO/FCPXML/Resolve XML export
  and import is higher value than building transitions/titles/A2 from scratch, and
  it matches how the community finishes films. `Render-AI-Code/timeline` and
  several open film-studio projects (Gobo, Spritz, Framebreaker) already treat XML
  export as the handoff.
- **ComfyUI is the extension surface.** Rather than implementing every new model,
  expose a ComfyUI workflow import/export or a bridge (submit a workflow, watch its
  progress, ingest outputs). The Studio can stay the shot/take/review layer while
  ComfyUI owns experimental graphs.
- **Camera/blocking from Blender** is an emerging workflow (Spritz). A glTF/USD/OTIO
  camera import is a natural fit for the existing blocking lens.
- **Audio-native models** (LTX-2.5, H3) change the review unit: a take includes
  synchronized audio; captions/transcripts and a voice lane (Qwen3-TTS) become
  first-class.

## 5. Hardware fit

| VRAM | Viable today | Recommended path |
|---|---|---|
| 8 GB | Wan 2.2 TI2V-5B with ComfyUI offload (experimental); Flux 2 Q4 + quantized/offloaded text encoder; LTX-2.5 int8 is not an official target | Wan 5B + Flux Q4 |
| 12 GB | Wan 5B; Flux Q4; LTX-2.5 int8 experimental | Wan 5B |
| 16 GB | Wan 5B; Flux Q4; LTX-2.5 int8/W4A8 experimental | Wan 5B |
| 24 GB | Wan 5B official; Wan 14B fp8 at 640²; Qwen-Image-Edit-2511 FP8; Hunyuan 1.5 (license) | Wan 5B/14B |
| 32–48 GB | LTX-2.5 bf16/FP8 official; Wan 14B 480p; Flux full | LTX-2.5 |
| 80 GB | Wan A14B 720p; LTX-2.5 bf16/4K | LTX-2.5 |
| Mac 64–128 GB | LTX-2 19B (current); LTX-2.5 bf16/MLX q8 | LTX-2.5 MLX when available |

## 6. Revised decisions

1. **Do not bolt CPU offload onto the vendored LTX-2 pipeline.** Adopt the official
   diffusers LTX-2.5 path, which already has `enable_model_cpu_offload()` and VAE
   tiling. Keep the vendored LTX-2 engine as a fallback during migration.
2. **Add Wan 2.2 TI2V-5B as the permissive low-VRAM engine.** Apache 2.0, diffusers,
   and a 5B model that fits consumer cards. Wan 2.2 S2V/Animate come later.
3. **Do not bundle license-restricted models.** HunyuanVideo 1.5 (EU/UK/KR),
   MiniMax H3 (US/EU/UK/KR), and Qwen-Image 2.1 (research-only) are optional
   user-supplied engines with an explicit license acknowledgement.
4. **Prioritize OTIO/XML and ComfyUI interop over NLE feature parity.** The
   finishing NLE already exists; the studio's value is the shot/take/review layer
   and a clean handoff.
5. **Add Qwen3-TTS (Apache 2.0) and SeedVR2 (Apache 2.0)** as optional post lanes
   before building bespoke audio/upscale features.
6. **Image lane:** keep Flux 2 for now; add GGUF for low VRAM; then split to
   **Z-Image Turbo** (fast, Apache 2.0) and **Qwen-Image-Edit-2511** (editing,
   Apache 2.0). Qwen-Image 2.1 is research-only and is not adopted.

## Sources

- LTX-2.5 / LTX-2 release notes: <https://github.com/Lightricks/LTX-Video>,
  <https://huggingface.co/Lightricks/LTX-2.5>,
  <https://huggingface.co/Lightricks/LTX-2.5-Diffusers>
- LTX-2.5 community license: <https://ltx.io/model/license>
- Wan 2.2: <https://github.com/Wan-Video/Wan2.2> (Apache 2.0; 5B/A14B,
  S2V, Animate; diffusers/ComfyUI)
- HunyuanVideo 1.5: <https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5>
  (community license excludes EU/UK/KR; 8.3B; ~14 GB with offload)
- MiniMax H3: <https://huggingface.co/MiniMaxAI/MiniMax-H3> (community license
  excludes US/EU/UK/KR; 33B; FreeVideo 8 GB)
- Qwen-Image 2.1: <https://huggingface.co/Qwen/Qwen-Image-2.1> (Qwen Research
  License; 7B unified generation/editing)
- Qwen3-TTS: <https://github.com/QwenLM/Qwen3-TTS> (Apache 2.0)
- SeedVR2: <https://github.com/ByteDance-Seed/SeedVR> and the official ComfyUI
  nodes (Apache 2.0)
- Z-Image Turbo: <https://github.com/Tongyi-MAI/Z-Image> (Apache 2.0; 6B; 8-step;
  MLX variant)
- Qwen-Image-Edit-2511: <https://huggingface.co/Qwen/Qwen-Image-Edit-2511>
  (Apache 2.0; 20B; FP8/GGUF)
- ComfyUI: <https://github.com/comfyanonymous/ComfyUI>; LTX-2.5 workflow:
  <https://comfy.org/ltx-2.5/>
- OpenTimelineIO: <https://opentimelineio.readthedocs.io/>; OTIO import/export in
  Resolve/Premiere/FCP is production-supported
- AA video leaderboard: <https://artificialanalysis.ai/video/leaderboard/text-to-video>
