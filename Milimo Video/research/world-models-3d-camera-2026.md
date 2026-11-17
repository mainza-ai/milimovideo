---
type: research
created: 2026-10-09
updated: 2026-10-09
tags: world-models 3d camera-control segmentation depth research
sources:
  - research survey 2026-10-09 (web)
related:
  - ../models/sam-3.md
  - ../decisions/sam3-microservice.md
  - ../pipelines/inpainting.md
  - ../roadmap/v2-roadmap.md
---

# Research: World Models, 3D, Camera Control, Segmentation/Tracking (Oct 2026)

Survey of adjacent capabilities that could become new Milimo Video workflows. All
post-Jan-2026 facts are web-sourced; unconfirmed items are marked.

## Landscape

**World models.** LTX-2.5 is now positioned as an open-weights world model (native
multishot continuity, prompt-controlled camera, robotics checkpoint). Playable open
world models are arriving on consumer GPUs: **ABot-World-0** (720p/16fps on a single
RTX 5090, ~19GB VRAM), **Zing-0.5** (Sep 2026, joint text+keyboard control, causal KV
cache, single-GPU real-time claimed), **DreamX-World 5B** (partially open, camera-
indexed revisits relevant to scene consistency). Licenses for the newest ones are
unconfirmed — watch, don't adopt yet.

**3D generation.** The 2026 open stack: **TRELLIS.2** (Microsoft, **MIT**, single
image→mesh with PBR GLB, multi-view up to ~16 views; ~24GB VRAM, community Mac
Metal port) as primary; **Hunyuan3D-2.1** (Tencent, best-in-class open PBR texturing
but **territorial license excludes EU/UK/KR**) as texture specialist; **TripoSG**
(VAST-AI, **MIT**, 8–12GB VRAM, shape-only) as low-VRAM fallback. Tripo P2.0 (Oct
2026, native quad meshes) is commercial — watch.

**Camera control.** No shipping open video model accepts exact per-frame camera
extrinsics natively (absence of evidence across targeted searches). The practical
2026 answer: **LTX-2.5's prompt-specified camera behavior** (dolly/pan/jib/rack
focus) + start/end keyframes. Precise-trajectory methods (AC3D Plücker tokens,
training-free CamTrol, SRENDER Jan 2026, NavCrafter Apr 2026) are research code —
watch as an adapter layer.

**Depth/geometry.** **Depth Anything 3** (ByteDance Seed, Nov 2025) is the new open
geometry SOTA: any-view depth + camera pose + metric depth in one model, ~36% better
pose accuracy than VGGT per their benchmark. **Small/Base are Apache-2.0** (Giant/Large
are CC BY-NC). Direct replacement for any depth stage.

**Segmentation/tracking → KEEP SAM 3.** Meta SAM 3 (Nov 19 2025, ~840M params) remains
the latest Segment Anything release — **no SAM 4 found as of Oct 2026**. It is the
SOTA foundation for exactly the microservice's job (text/click/box → segment →
track), already integrated, nothing in 2026 beats it by a rewrite-justifying margin.
Companion **SAM 3D Objects/Body** (same SAM license) does single-image→textured mesh
— a natural new previz capability sharing the SAM 3 front end. Evaluate **sam3.cpp**
(2026 ggml port, license unconfirmed) as a lighter Apple-Silicon inference path.

**Video inpainting/outpainting.** **ProPainter** (MIT, actively maintained, chunked
FP16, modest GPUs, ships a FastAPI server + Web UI) is the practical workhorse for
object removal/completion — adopt as a dedicated lane alongside the Flux-based
editor. **Netflix VOID** (Apr 2026, **Apache-2.0**, ECCV 2026) is the quality leader
for interaction-aware deletion (shadows, reflections, physical consequences) but needs
**40GB+ VRAM** — watch. **LTX-2.3 IC-LoRA in/outpainting** works within the existing
LTX stack (evaluate). OmnimatteZero, DiffuEraser, Seen-to-Scene (CVPR 2026),
HL-OutPaint (SIGGRAPH '26): watch.

## Comparative Assessment

| Capability | Pick | License | Local VRAM | Verdict |
|---|---|---|---|---|
| Segmentation/tracking | SAM 3 (keep) | SAM License (commercial OK) | ~3.4GB weights | **Keep** — no successor |
| 3D previz/assets | TRELLIS.2 | **MIT** | ~24GB (16GB fp16) | **Adopt** — primary |
| 3D low-VRAM fallback | TripoSG | **MIT** | 8–12GB | **Adopt** — fallback |
| 3D textures | Hunyuan3D-2.1 | excludes EU/UK/KR | 10–21GB | Adopt w/ license caveat |
| Depth/pose | Depth Anything 3 S/B | **Apache-2.0** | <12GB (streaming) | **Adopt** |
| Camera control | LTX-2.5 prompts+keyframes | LTX Community | as video slot | **Adopt** — practical |
| Video object removal | ProPainter | **MIT** | modest GPUs | **Adopt** — workhorse |
| Interaction-aware deletion | Netflix VOID | **Apache-2.0** | 40GB+ | Watch — server-class |
| World models (playable) | Zing-0.5 / ABot-World-0 | unconfirmed | 19GB+ | Watch — licenses TBD |

## Relevance to Milimo Video

Keep the SAM 3 microservice (add SAM 3D Objects as a prop-previz capability; spike
sam3.cpp for Mac). Add three new capability lanes in v2: (1) **3D previz** via
TRELLIS.2 (image→PBR mesh) feeding concept art and set design; (2) **geometry** via
Depth Anything 3 (Apache variants) for depth-aware effects and camera planning;
(3) **dedicated video object removal** via ProPainter alongside the generative
inpaint editor. Camera control comes free with the LTX-2.5 upgrade
(prompt + keyframes); precise-trajectory adapters stay on the watch list.

## Sources

- LTX-2.5 world model: https://ltx.io/model/capabilities/world-model · https://www.marktechpost.com/2026/08/11/the-video-production-stack-now-fits-on-one-desk-ltx-2-5-launches-as-nvidia-accelerated-open-weights-world-model/
- 3D: https://triposr.org/blog/hunyuan3d-vs-trellis · https://github.com/kurplee/image-to-3dlab · https://3dprintingdaily.com/news/native-quad-meshes-arrive-in-ai-3d-generation-with-tripo-p20
- Camera: https://arxiv.org/abs/2601.09697v1 (SRENDER) · https://arxiv.org/pdf/2604.02828v1.pdf (NavCrafter) · https://arxiv.org/html/2411.18673v4 (AC3D)
- Depth: https://github.com/ByteDance-Seed/Depth-Anything-3
- SAM 3: https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/ · https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/
- Inpainting: https://github.com/Netflix/void-model · https://github.com/zhenbah/propainter · https://github.com/dvirsamuel/OmnimatteZero

Unconfirmed: Zing-0.5/ABot-World-0/OmnimatteZero/sam3.cpp/SRENDER/NavCrafter licenses;
TRELLIS.2 and Tripo P2.0 exact release dates; MPS performance numbers for SAM 3/VOID;
no-SAM-4 is absence-of-evidence (~6 targeted searches), not proof.

## Related

- [[../models/sam-3.md]] — the microservice this verdict keeps
- [[../decisions/sam3-microservice.md]] — why it's isolated
- [[../pipelines/inpainting.md]] — the current inpaint path
- [[../roadmap/v2-roadmap.md]] — the v2 plan
