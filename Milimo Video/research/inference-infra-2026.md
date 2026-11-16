---
type: research
created: 2026-10-09
updated: 2026-10-09
tags: inference quantization vram apple-silicon mlx mps research
sources:
  - research survey 2026-10-09 (web)
related:
  - ../concepts/memory-slots.md
  - ../concepts/mps-first.md
  - ../systems/architecture.md
  - ../roadmap/v2-roadmap.md
---

# Research: Inference Engines, Quantization & Local-First Infra (Oct 2026)

Survey of the 2026 inference/infra landscape and what it means for the
MemoryManager slot architecture and the MPS-first engineering law. All
post-Jan-2026 facts are web-sourced; unconfirmed items are marked.

## Landscape

**Diffusion serving has grown up.** Three production options now serve video diffusion:
- **vLLM-Omni** (`vllm-project/vllm-omni`, container v0.26.0, Sep 2026) — purpose-built
  vLLM fork for diffusion/image-video (Flux, Wan2.2, LTX-2, Qwen-Image, GLM-Image,
  Cosmos3, MiniMax-H3) with ModelOpt FP8/NVFP4/MXFP8, sequential *and* layerwise
  blockwise CPU offload, multi-GPU. Notably, it independently re-derived Milimo's
  two mechanisms: model-level mutual exclusion (DiT↔encoder) and layerwise offload.
- **SGLang-diffusion** (`sglang.multimodal_gen`, v0.5.12–0.5.14, Jun–Sep 2026) — 20+
  models incl. **LTX-2/2.3/2.5**, Wan, FLUX, HunyuanVideo; NVIDIA/AMD/Intel/Ascend
  **plus Apple Silicon (M-series via MPS)**; TP + Ulysses/Ring sequence parallelism;
  dynamic request batching; ModelOpt NVFP4; OpenAI-compatible API. Best fit for
  Milimo's stack on paper.
- **TensorRT-LLM VisualGen** (TRT-LLM 1.3.0, mid-2026) — first-class diffusion
  serving (FLUX.1/2, Wan 2.1/2.2, LTX-2) with FP8/NVFP4, TeaCache, multi-node
  workers, pre-quantized `nvidia/` Wan2.2 checkpoints. **NVIDIA-only** — irrelevant
  to MPS-first.
- **xDiT** — open parallel-inference *library* for DiTs (Ulysses/Ring/PipeFusion,
  CFG-parallel, FSDP2, torchao FP8/FP4 GEMMs; LTX-2.5 support added ~Aug 2026).
  Multi-GPU/multi-machine via torchrun. **ComfyUI v0.38.0** (Sep 2026) remains the
dominant local-first runtime and the reference implementation for memory management,
  GGUF/Nunchaku/FP8 loaders, and Apple-Silicon execution paths.

**Apple Silicon 2026.** **MLX now runs video DiTs**: community ports for LTX-2 19B
(full 48-layer DiT port, Mar 2026), LTX-2.3 22B with int4/int8 quantized inference,
MiniMax-H3 33B. WWDC26 announced production MLX distributed inference (JACCL RDMA
over Thunderbolt 5) — but **LLM-only; image/video gen explicitly lacks multi-node
sharding**. **PyTorch MPS 2.13/2.14** (Jul–Sep 2026): FlexAttention landed (~12× on
sparse patterns), an 8.5×-slow GEMV routing bug fixed, native Metal SVD/eigh/QR,
2–4× prefill attention gains, **bfloat16 reliable since 2.10**. **FP8 still
unsupported on MPS** (Metal lacks 8-bit float). CPU VAE decode remains the safety
valve — still the slowest/riskiest op for video DiTs on MPS, and MPS op bugs persist
(e.g., a `baddbmm` NaN bug, Sep 2026).

**Quantization for video diffusion.** FP8 is productionized (comfy-quants LTX-2.3
family; BFL+NVIDIA official FP8/NVFP4 for all FLUX.2 Klein: FP8 ≈ 1.6× faster,
−40% VRAM; NVFP4 ≈ 2.7×, −55% VRAM, RTX-only). LTX-2 19B: ~32GB BF16 → ~16GB FP8
→ ~10GB NVFP4. INT4 (Nunchaku/SVDQuant): confirmed for image models (FLUX,
Qwen-Image, Z-Image) — **video DiT coverage unconfirmed**. **GGUF now covers video**
(QuantStack Wan2.2 GGUF, LTX-2 Q4_K_M ≈ 12GB file; ComfyUI-GGUF node,
ollamadiffuser, stable-diffusion-cpp with Metal). DeltaQuant (CVPR 2026)
research-validates 4-bit video DiTs (Wan2.2, LTX-Video) with quality holding.
On Apple Silicon: no FP8/NVFP4 execution; INT4 works via MLX `quantized_matmul` in
community ports.

**VRAM management.** The slot concept is **validated, not obsoleted** — vLLM-Omni and
ComfyUI's `model_management` converged on the same pattern. diffusers 0.39 (Aug
2026) still offers model/sequential CPU offload (+VAE tiling): the classic ladder
44GB → 25GB → 22GB → 7GB. **No 2026 technique lets a 19B DiT + 9B image model + 12B
LLM all sit resident on 16GB VRAM / 32GB unified memory** — hot-swap got faster
(pinned memory, async prefetch) but no sub-second multi-GB swaps. **Quantization is
the actual lever** that changes slot math (FP8 halves slots; NVFP4 quarters them —
CUDA-only). NVMe weight streaming for DiTs: no 2026 ecosystem (LLM/KV-cache story
only). LAN multi-machine video diffusion: no practical prosumer story (Ethernet
latency kills per-step collectives); multi-Mac MLX sharding is LLM-only.

**Packaging.** safetensors remains the container standard; **compressed-tensors**
(inside safetensors: FP8/INT4/NVFP4 + scales) is the HF-native quantized format
(auto-detected by SGLang/vLLM-Omni); **GGUF** is the community distribution standard
for quantized diffusion (builders like QuantStack/city96; HF transformers loads GGUF
natively since Sep 2026 via ggml Metal kernels); **MLX format** for Apple Silicon.
(Context: NVIDIA acquired Hugging Face, confirmed Sep 3 2026 — packaging politics to
watch, no format disruption observed.)

## Comparative Assessment

| Lever | Effect | Apple Silicon? | Verdict |
|---|---|---|---|
| FP8 transformer (NVIDIA) | −40–50% VRAM, ~1.6× faster | ❌ No (Metal lacks FP8) | **Adopt** on NVIDIA — collapses slots |
| NVFP4 (Blackwell) | −55% VRAM, ~2.7× faster | ❌ | Adopt where hardware allows |
| GGUF Q4–Q6 video DiTs | ~12GB file for LTX-2 | ✅ via Metal runtimes | **Adopt** as distribution format |
| INT4 (Nunchaku) | 3× faster, ~7GB for 12B | partial (MLX) | Watch for video DiTs |
| SGLang-diffusion backend | Server-grade LTX-2.x + MPS | ✅ MPS listed | **Evaluate** as optional NVIDIA backend |
| MLX DiT ports | int4/int8 video on Mac | ✅ native | **Spike** — the true Mac upgrade path |
| Multi-GPU (xDiT/SGLang TP) | scales 19B+ models | ❌ NVIDIA-only | Only for multi-GPU boxes |
| Multi-Mac / LAN | — | LLM-only | **Don't invest** |

## Relevance to Milimo Video

**Slot architecture: KEEP** — mutual exclusion is the industry-standard pattern
(re-derived by vLLM-Omni in 2026). Update slot budgets with FP8 variants; keep
sequential CPU offload as the layer-level escape hatch. **MPS-first law: KEEP with
one relaxation** — drop mandatory float32 transformer forcing now that bf16 is stable
on MPS ≥2.10; keep CPU VAE decode as default-safe; budget ~5–8× slower than an RTX
4090 for the video slot on Mac. Concrete 2026 moves: (1) official FP8 model variants
make 16GB-VRAM NVIDIA cards viable single-slot (buys nothing on MPS); (2) evaluate
**SGLang-diffusion as an optional NVIDIA backend** (native LTX-2.x, MPS platform,
ModelOpt quant) while keeping diffusers on MPS; (3) **MLX backend spike** for the
video slot on Apple Silicon — the only 2026 route to materially faster/bigger models
on Mac; (4) ship **GGUF + compressed-tensors** artifacts for quantized distribution;
(5) don't invest in multi-Mac, NVMe DiT streaming, vLLM-Omni (server-only, no MPS),
or TRT-LLM VisualGen (NVIDIA-only).

## Sources

- vLLM-Omni: https://github.com/vllm-project/vllm-omni · https://github.com/liunix61/vllm-omni/blob/HEAD/docs/design/architecture_overview.md
- SGLang: https://github.com/antgroup/sglang/blob/HEAD/docs/docs/sglang-diffusion/quantization.mdx · https://github.com/bytedance-iaas/sglang/blob/HEAD/python/sglang/multimodal_gen/README.md
- TRT-LLM: https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc14 · https://github.com/NVIDIA/TensorRT-LLM/pull/12464
- xDiT: https://github.com/xdit-project/xdit · ComfyUI: https://undercodenews.com/comfyui-v0380-a-major-ai-generation-upgrade-brings-qwen-image-21-gpt-6-claude-opus-55-seedream-50-and-serious-performance-gains-video/
- MLX: https://github.com/dgrauet/ltx-2-mlx/commit/dcd639e77b25a46c2b562463ed4c6f9cf16dccf5 · https://github.com/appautomaton/MLX-GenAI
- PyTorch MPS: https://github.com/meta-pytorch/torch-release-notes/blob/HEAD/2.13.0/final.md · http://ecorpit.com/pytorch-2-13-flexattention-apple-silicon-12x-speedup-2026/
- Quantization: https://github.com/comfy-org/comfy-quants/blob/HEAD/docs/quantization/ltx2.md · https://github.com/nunchaku-ai/nunchaku · http://openaccess.thecvf.com/content/CVPR2026/papers/Li_DeltaQuant_4-bit_Video_Diffusion_Models_with_Spatiotemporal_Delta_Smoothing_CVPR_2026_paper.pdf
- GGUF video: https://github.com/localkinai/ollamadiffuser/blob/HEAD/GGUF_GUIDE.md · https://dev.to/gary_yan_86eb77d35e0070f5/how-to-install-and-configure-ltx-2-gguf-models-in-comfyui-complete-2026-guide-1d3m
- Packaging: https://huggingface.co/docs/transformers/quantization/compressed_tensors

Unconfirmed: Nunchaku INT4/NVFP4 for video DiTs; whether CPU VAE decode is still
*required* vs merely safest on MPS; exact LTX-2.5 specs beyond family naming;
FLUX.2 Klein FP8 figures are community-reported; vllm-metal version details single-source.

## Related

- [[../concepts/memory-slots.md]] — the slot design this validates
- [[../concepts/mps-first.md]] — the law this partially relaxes
- [[../systems/architecture.md]] — the backend this may gain an engine option
- [[../roadmap/v2-roadmap.md]] — the v2 plan
