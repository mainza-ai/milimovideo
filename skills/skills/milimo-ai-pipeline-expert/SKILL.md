---
name: milimo-ai-pipeline-expert
description: Deep expertise on LTX-2 video pipelines, Flux 2 image/inpainting pipelines, and memory coordination on unified Apple Silicon hardware. Use this for debugging GPU memory issues (OOM), modifying tensor inputs, investigating chained generation (quantum alignment), or customizing inference steps. 
---

# Milimo AI Pipeline Expert Skill

As the Milimo AI Pipeline Expert, your domain covers the deeply integrated diffusion/transformer logic and VRAM management strategies for LTX-2, Flux 2, and the LLM enhancement endpoints.

## Device & Memory Constraints (CRITICAL)
- **Apple Silicon (MPS)** is the primary assumed target environment, though CUDA is supported.
- The `MemoryManager` (`memory_manager.py`) strictly enforces **mutual exclusion** between LTX-2 and Flux 2 to prevent unified memory OOM crashes.
- Before loading LTX-2, you must call `memory_manager.prepare_for("video")`.
- Before loading Flux 2, you must call `memory_manager.prepare_for("image")`.
- **MPS Bug Hacks**: Flux and LTX VAE decoders must run in `float32` on MPS, or they will output black frames. `torch.mps.empty_cache()` and `gc.collect()` must be manually executed between pipeline swaps.

## LTX-2 Video Pipeline (`model_engine.py` & `tasks/video.py`)
- LTX-2 is a 19B Dual-Stream Transformer. The distillation checkpoint (`ltx-2-19b-distilled`) runs at 8 inference steps.
- **Three Pipeline Types**:
  1. `ti2vid`: Default Text/Image-to-Video. Replaces noisy latents at frame 0 with input image conditioning.
  2. `ic_lora`: Used for subject consistency.
  3. `keyframe`: Used for start-to-end frame interpolation.
- **Prompt Enhancement**: By default, user prompts are piped through `llm.py` before hitting the text encoder.

### Chained Generation (`tasks/chained.py`)
- Standard LTX-2 supports max 505 frames.
- Chained gen uses **autoregressive chunking** (505 frames per chunk, 24 frame overlap). 
- **Quantum Alignment**: Latent space is 8 pixels per token. The 24-pixel overlap is mathematically quantized into `latent_slice_count`. Trimming happens via `ffmpeg -ss` to perfectly match the latent splice point, preventing "frozen anchor" visual artifacts during transition.

## Flux 2 Image & Inpainting (`models/flux_wrapper.py`)
- **FluxInpainter Singleton**: Holds the Flow model, AE, Qwen 3 text encoder, and CLIP ViT-L IP-Adapter.
- **AE Hot-Swapping**: Can toggle between the Native AutoEncoder (supports temporal reference offsets for image conditioning) and the diffusers wrapper fallback.
- **Sequential True CFG**: Flux 2 is distillation-guided, ignoring negative prompts. Milimo implements a custom 2-pass CFG loop inside `denoise_inpaint()` to restore negative prompt capability (at the cost of double inference time).
- **Inpainting (RePaint)**: Uses mask interpolation merging `x_pred` and `x_known` at each timestep.

## Ollama Coordination (`llm.py`)
- Prompts enhanced via local VLM running alongside generation.
- **KEEP_ALIVE Management**: To prevent VRAM lockout, `llm.py` appends `keep_alive: 0` to Ollama requests based on user settings, explicitly unloading the language model before returning the enhanced text to the video caller.
