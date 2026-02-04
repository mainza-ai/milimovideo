# LTX-2 System Documentation

## 1. Executive Summary

**System Name**: LTX-2 Audio-Video Generation System

**Description**:
LTX-2 is a state-of-the-art, DiT (Diffusion Transformer) based foundation model designed for simultaneous generation of high-fidelity video and synchronized audio. Unlike distinct pipelines that generate video then audio (or vice versa), LTX-2 models both modalities jointly using an asymmetric dual-stream transformer architecture.

**Core Capabilities**:
- **Joint Generation**: Synthesizes synchronized video (14B params) and audio (5B params).
- **Multi-Stage Inference**: Supports fast distilled inference and high-quality two-stage generation with upsampling.
- **Fine-tuning**: Comprehensive training suite for LoRA, full fine-tuning, and IC-LoRA (video-to-video).
- **Conditioning**: Supports text-to-video, image-to-video, and video-to-video workflows.

**Intended Users**:
- AI Researchers advancing multimodal generative models.
- ML Engineers deploying high-fidelity video generation services.
- Creative Professionals requiring fine-grained control over video content creation.

## 2. System Architecture Overview

**Architectural Style**: Monorepo with modular packages.
- **Core (ltx-core)**: Foundation models, diffusion components (schedulers, guiders), and base logic.
- **Inference (ltx-pipelines)**: High-level orchestration of generation workflows.
- **Training (ltx-trainer)**: Tools for model fine-tuning and adaptation.

**Key Components**:
1.  **Dual-Stream Transformer**: The heart of the system. An asymmetric transformer with bidirectional cross-attention between a Video Stream (14B parameters, 3D RoPE) and an Audio Stream (5B parameters, 1D RoPE).
2.  **Gemma 3 Text Encoder**: Uses a frozen LLM (Gemma 3) with a learnable "Thinking Token" connector to generate separate semantic embeddings for video and audio.
3.  **Variational Autoencoders (VAEs)**:
    - **Video VAE**: Spatiotemporal compression of pixel data to latents.
    - **Audio VAE**: Compresses mel-spectrograms to latents.
4.  **Vocoder**: Converts audio latents/spectrograms back to waveforms.
5.  **Upsampler**: A spatial upsampler model used in two-stage pipelines to increase resolution (e.g., from 384p to 768p).

**Runtime Environment**:
- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Package Manager**: `uv` (recommended) or `pip`
- **Hardware**: NVIDIA GPUs (80GB+ VRAM recommended for training, scaling down for inference with FP8/Quantization).

## 3. Repository Structure

The repository is organized as a Python monorepo using `uv` workspaces.

```text
/
├── packages/
│   ├── ltx-core/          # Foundational modeling code
│   │   ├── src/ltx_core/
│   │   │   ├── components/  # Diffusion building blocks (Schedulers, Guiders)
│   │   │   ├── conditioning/ # Input conditioning logic
│   │   │   ├── guidance/    # Perturbation logic (STG)
│   │   │   ├── loader/      # Checkpoint loading & LoRA fusion
│   │   │   ├── model/       # PyTorch Model Definitions (Transformer, VAEs)
│   │   │   └── text_encoders/ # Gemma 3 integration
│   ├── ltx-pipelines/     # Inference workflows
│   │   ├── src/ltx_pipelines/
│   │   │   ├── ti2vid_two_stages.py  # Production T2V/I2V pipeline
│   │   │   ├── ti2vid_one_stage.py   # Reference pipeline
│   │   │   ├── ic_lora.py            # Video-to-Video pipeline
│   │   │   └── distiled.py           # Fast inference pipeline
│   ├── ltx-trainer/       # Training framework
│   │   ├── src/ltx_trainer/
│   │   │   ├── trainer.py     # Main training loop
│   │   │   └── training_strategies/ # LoRA, Full Finetune, etc.
│   │   ├── configs/       # Training YAML configs
│   │   └── scripts/       # Training entry points
├── README.md              # Project entry point
├── pyproject.toml         # Root workspace config
└── uv.lock                # Dependency lockfile
```

## 4. Core Concepts & Design Philosophy

**Asymmetry**: The model acknowledges that video is informationally denser than audio. It allocates ~3x more parameters to the video stream while maintaining tight synchronization via cross-attention.

**Latent Diffusion**: All generation happens in a compressed latent space to ensure computational efficiency.

**Modular Diffusion**:
- **Schedulers**: Control the noise schedule (Linear, Cosine, LTX2-specific).
- **Guiders**: Encapsulate logic for CFG (Classifier-Free Guidance) and STG (Spatio-Temporal Guidance).
- **Noisers**: Manage noise injection.
- This decoupling allows easy experimentation with different sampling strategies without rewriting the model loop.

**Two-Stage Generation**:
- **Stage 1 (Generation)**: Creates the content structure at lower resolution.
- **Stage 2 (Refine & Upscale)**: Upscales the latent and refines details, often using a distilled model or specialized LoRA for sharpness.

## 5. Detailed Module & File Documentation

### 5.1 `ltx-core` Package
*Location: `packages/ltx-core/`*
*Purpose: Foundational library containing model definitions and shared components.*

#### Core Model Components
- **`src/ltx_core/model/transformer/model.py`**:
    - **Class `LTXModel`**: The main PyTorch module. Initializes the Video and Audio streams, embedding handling, and the stack of `BasicAVTransformerBlock`. Handles the forward pass involving patchification, transformer execution, and unpatchification.
    - **Class `X0Model`**: A wrapper that converts model "velocity" predictions into denoised "X0" predictions (latents).

- **`src/ltx_core/model/transformer/transformer.py`**:
    - **Class `BasicAVTransformerBlock`**: accessible unit of the transformer. Contains:
        1. Self-Attention (Video & Audio independent)
        2. Text Cross-Attention (Conditioning)
        3. A↔V Cross-Attention (Syncs modalities using 1D temporal RoPE)
        4. Feed-Forward Networks (FFN)

- **`src/ltx_core/components/schedulers.py`**:
    - **Class `LTX2Scheduler`**: The default scheduler. Implements a sigma schedule that shifts based on token count (resolution dependent) and stretches to a terminal value.
    - **Class `LinearQuadraticScheduler`**: Hybrid schedule (linear start, quadratic tail).

- **`src/ltx_core/components/guiders.py`**:
    - **Class `MultiModalGuider`**: Sophisticated guider that combines:
        - `cfg_scale`: Text adherence.
        - `stg_scale`: Spatio-Temporal Guidance (perturbation-based structure preservation).
        - `modality_scale`: Audio-Video sync enforcement.
    - **Class `CFGGuider`**: Standard `(cond - uncond) * scale` logic.

### 5.2 `ltx-pipelines` Package
*Location: `packages/ltx-pipelines/`*
*Purpose: User-facing generation workflows.*

#### Pipelines
- **`src/ltx_pipelines/ti2vid_two_stages.py`**:
    - **Class `TI2VidTwoStagesPipeline`**: The standard high-quality pipeline.
        - **Step 1**: Generates 1/2 resolution video using `LTX2Scheduler` and `MultiModalGuider`.
        - **Step 2**: Upscales latents using `spatial_upsampler`, then denoises using a Distilled LoRA at high resolution for sharpness.
    - **Function `__call__`**: Entry point accepting `prompt`, `images` (for I2V), and guidance params.

- **`src/ltx_pipelines/utils/helpers.py`**:
    - **Function `euler_denoising_loop`**: The core loop iterating over sigma steps. Calls the denoise function (model + guider) and steps the latent state.
    - **Function `gradient_estimating_euler_denoising_loop`**: An optimized loop using velocity gradient estimation to reduce step count (e.g., from 40 to 20).

### 5.3 `ltx-trainer` Package
*Location: `packages/ltx-trainer/`*
*Purpose: Fine-tuning and training.*

#### Core Training Logic
- **`src/ltx_trainer/trainer.py`**:
    - **Class `LtxvTrainer`**: Manages the training lifecycle.
        - Handles `Accelerate` setup (DDP/FSDP).
        - Manages checkpoints (loading `safetensors`, saving).
        - Runs the training loop: Fetch batch -> Forward -> Loss -> Backward -> Optimizer Step.
        - Runs validation sampling periodically (generating videos to track progress).

- **`src/ltx_trainer/config.py`**:
    - Defines the `LtxTrainerConfig` Pydantic model, mapping directly to the YAML configuration files.

- **`src/ltx_trainer/training_strategies/`**:
    - Contains logic for different training modes (e.g., standard LoRA vs. IC-LoRA).
    
    - **`ltx-trainer/docs/configuration-reference.md`**:
        - A critical reference file detailing all valid YAML configuration options.
        - **Important**: It details the `target_modules` for LoRA training, differentiating between video-only (`attn1`, `ff`), audio-only (`audio_attn1`, `audio_ff`), and cross-modal (`audio_to_video_attn`) modules.

#### CLI Tools & Scripts (`packages/ltx-trainer/scripts/`)
- **`process_dataset.py`**:
    - **Purpose**: Pre-computes latents and text embeddings to accelerate training.
    - **Features**:
        - Supports resolution bucketing (e.g., `768x768x49`).
        - Handles reference videos for IC-LoRA (scaling them down if needed).
        - Can decode latents back to video for verification.
    - **Usage**: `python scripts/process_dataset.py dataset.json ...`

- **`process_captions.py`**:
    - **Purpose**: Specialized script for cleaning and embedding text captions.
    - **Features**:
        - Automatic LLM prefix removal (e.g., "In this video...").
        - Supports batch processing (batch size 1 for now due to Gemma tokenizer limitations).
    - **Usage**: `python scripts/process_captions.py dataset.json ...`

- **`inference.py`**:
    - **Purpose**: The main entry point for CLI-based generation.
    - **Modes**:
        - **T2V**: Text-to-Video (default).
        - **I2V**: Image-to-Video (`--condition-image`).
        - **V2V**: Video-to-Video (`--reference-video`).
        - **V2V+I2V**: Combined mode.
    - **Key Args**: `--stg-scale` (Structure guidance), `--stg-mode` (audio-video or video only), `--skip-audio`.

## 6. Data Flow & Control Flow

### Inference Data Flow (Two-Stage Pipeline)

1.  **Input**: User provides `Prompt` ("A cat jumping"), `Image` (Optional start frame), and Config (Resolution, Steps).
2.  **Text Encoding**:
    - Prompt passed to `GemmaTextEncoder`.
    - Returns `VideoContext` (4096 dim) and `AudioContext` (2048 dim) embeddings.
3.  **Latent Initialization**:
    - Random Gaussian noise created for Video (`[B, C, F, H/32, W/32]`) and Audio.
    - If Image provided: Image is encoded via VAE -> replaces the first noise frame (latent replacement).
4.  **Stage 1 Denoising (Low Res)**:
    - Loop `t` from 1.0 to 0.0:
        - `Model(Latent_t, Context)` -> `Velocity`.
        - `Guider` modifies `Velocity` (CFG/STG).
        - `Stepper` computes `Latent_{t-1}`.
5.  **Upscaling**:
    - Stage 1 Output Latents -> `SpatialUpsampler` -> 2x Resolution Latents.
6.  **Stage 2 Denoising (High Res)**:
    - Loop with fewer steps (e.g., 8-10) using Distilled LoRA.
    - Refines details.
7.  **Decoding**:
    - Video Latents -> `VideoVAE Decoder` -> MP4 frames.
    - Audio Latents -> `AudioVAE Decoder` -> Spectrogram -> `Vocoder` -> WAV.
8.  **Output**: Combined video file.

## 7. Configuration & Environment Variables

**Training Configuration**:
- Uses YAML files (e.g., `configs/ltx2_av_lora.yaml`).
- **Key Params**:
    - `model.model_path`: Path to base checkpoint.
    - `lora.rank`: LoRA adapter size (e.g., 64).
    - `optimization.batch_size`: Per-device batch size.
    - `optimization.learning_rate`: e.g., 0.0001.
    - **LoRA Target Modules**:
        - **Video-Only**: `["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2"]`
        - **Audio-Video Shared**: `["to_k", "to_q", "to_v", "to_out.0"]` (matches all attention branches including cross-modal).

**Inference Configuration**:
- Typically passed via CLI args or Python Class `init`.
- **Environment Variables**:
    - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Vital for memory management on fragmentation-prone workloads.

## 8. External Integrations

- **Gemma 3**: Critical dependency for text understanding. The repo assumes a local copy or HuggingFace hub download of Gemma 3.
- **HuggingFace Hub**: For model weights and dataset interaction.
- **W&B (Weights & Biases)**: Integrated for training experiment tracking (loss curves, validation samples).

## 9. Extension & Customization Guide

### Adding a New Scheduler
1. Implement `SchedulerProtocol` in `ltx-core/components/protocols.py`.
2. Add your class to `ltx-core/components/schedulers.py`.
3. Register it in the pipeline or config.

### Customizing Training
1. Create a new strategy in `ltx-trainer/training_strategies/`.
2. Inherit from `BaseTrainingStrategy`.
3. Implement `computer_loss()` and `prepare_training_inputs()`.

## 10. Debugging & Maintenance Guide

**Common Failure Points**:
- **OOM (Out of Memory)**:
    - *Fix*: Enable `fp8transformer=True`, reduce batch size, or use `PYTORCH_CUDA_ALLOC_CONF`.
- **Resolution Mismatch**:
    - *Fix*: Ensure H/W are divisible by 32 (One-Stage) or 64 (Two-Stage). `assert_resolution` in helpers enforces this.
- **Missing Weights**:
    - *Fix*: Ensure all parts (VAE, Upscaler, Gemma) are downloaded.

**Testing**:
- Uses `pytest`. Run `uv run pytest` from root.

## 11. Known Risks, Limitations & Technical Debt

- **Monorepo Complexity**: Dependencies between packages are managed via `uv`. Mixing `pip install` with `uv sync` can break the environment.
- **Memory Hungry**: Full fine-tuning requires significant VRAM (80GB+). Consumer cards are limited to LoRA/Quantized training.
- **Inference Speed**: Two-stage generation is slow. Use `DistilledPipeline` for speed-critical apps.
- **Gemma dependency**: heavily relies on specific Gemma implementation; upgrades to Gemma might break the text encoder connector.

## 12. Deep Dive: Internal Mechanics

### 12.1 Prompt Engineering & System Prompts
The system enforces a strict "Creative Assistant" persona via system prompts found in `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/`.
- **Guidelines**:
    - Use present-progressive verbs ("is walking").
    - Audio descriptions must be integrated chronologically ("Ambient cafe sounds fill the space").
    - **Forbidden**: Non-visual senses (smell, taste), dramatic/exaggerated terms ("blinding light").
    - **Format**: `Style: <style>, <rest of prompt>.`

### 12.2 Conditioning Internals
- **Keyframe Conditioning** is implemented in `KeyframeCond` (`ltx_core/conditioning/types/keyframe_cond.py`).
    - It works by **masking**: A `denoise_mask` is created where the keyframe pixels are set to `1.0 - strength`.
    - This effectively locks the keyframe pixels during the diffusion process based on the strength parameter.

### 12.3 Spatial Upsampler
- Located in `packages/ltx-core/src/ltx_core/model/upsampler/spatial_rational_resampler.py`.
- Uses a **Rational Resampling** strategy:
    - **Upsample**: Learned `PixelShuffle` (e.g., scale 2.0).
    - **Downsample**: Fixed stride-based `BlurDownsample` for anti-aliasing.
    - Supports specific rational scales: 0.75, 1.5, 2.0, 4.0.

### 12.4 Guidance & STG
- **STG (Spatio-Temporal Guidance)** works by selectively **skipping attention mechanisms** in specific blocks.
- Defined in `perturbations.py`, types include:
    - `SKIP_A2V_CROSS_ATTN`: Decouples audio influence on video.
    - `SKIP_VIDEO_SELF_ATTN`: Breaks temporal/spatial coherence to guide structure.
- This "perturbation" creates a negative signal that the model steers away from (similar to CFG's unconditional negative signal).

### 12.5 Advanced LoRA Fusion
- **Float8 Support**: The loader (`fuse_loras.py`) includes custom Triton kernels (`fused_add_round_kernel`) to handle `float8` weights efficiently.
- **Fusion Logic**: New weights are calculated as `Base + (LoRA_B @ LoRA_A) * Scale`. The system handles precision casting automatically to avoid degradation.

### 12.6 Pipeline Architecture & Model Ledger
- **Model Ledger** (`ltx_pipelines/utils/model_ledger.py`):
    - Acts as a **central factory** for all model components.
    - **No Caching**: Models are instantiated on-demand to manage GPU memory aggressively.
    - Handles LoRA composition via `with_loras()`, creating lightweight copies of the ledger with new adapter configurations.
- **Keyframe Interpolation**:
    - Implemented in `keyframe_interpolation.py`.
    - **Two-Stage Process**: 
        1. Generates low-res video conditioned on keyframes. 
        2. Upscales and refines using a generic `upsample_video` and a second denoising loop.

## 13. Super Deep Dive: Internal Mechanics & Algorithms

### 13.1 Mathematical Implementation Details
- **LTX2Scheduler Math** (`schedulers.py`):
    - Uses a **token-dependent shifting** mechanism.
    - Formula: `sigma_shift = (tokens) * mm + b` where `mm` and `b` are derived from base/max anchors (1024, 4096).
    - **Stretching**: The final schedule is stretched to force the last sigma to match a `terminal` value (default 0.1), preventing the model from generating pure noise at the end.
- **Attention Hierarchy** (`attention.py`):
    - The system dynamically selects the best available attention implementation:
        1.  **FlashAttention3**: Highest priority, if installed.
        2.  **xFormers**: Memory-efficient attention, if FA3 is missing.
        3.  **PyTorch**: Standard `scaled_dot_product_attention` as fallback.

### 13.2 VAE Tiling & Sampling Internals
- **Trapezoidal Blending** (`tiling.py`):
    - Tiling is not just a hard cut. It uses **1D trapezoidal masks** with linear ramps (`linspace`) for feathering.
    - For N-dimensional tensors, 1D masks are broadcast and multiplied to create a smooth N-D blending window.
- **Sampling Tricks** (`sampling.py`):
    - `SpaceToDepthDownsample`: When stride is 2, it duplicates the first frame (`x[:, :, :1, :, :]`) before concatenation. This padding preserves **causal temporal consistency**.

### 13.3 Transformer Components
- **3D RoPE** (`rope.py`):
    - Implements **3D Rotary Positional Embeddings** to handle (Time, Height, Width) dimensions simultaneously.
    - Supports two modes:
        1.  `INTERLEAVED`: Standard rotary application.
        2.  `SPLIT`: Splits dimensions for specific optimizations.
    - **Frequency Grid**: Pre-computes and caches frequency grids to avoid re-calculation overhead (`lru_cache`).
- **AdaLN-Single** (`adaln.py`):
    - Based on **PixArt-Alpha** architecture (`AdaLayerNormSingle`).
    - Uses a combined **Timestep + Size Embedding** approach. The timestep embedding is projected and then fused with resolution/aspect-ratio embeddings to condition the normalization layers.

### 13.4 Training Strategy Differentiation
- **Timestep Sampling** (`timestep_samplers.py`):
    - **Shifted Logit-Normal**: The diffusion noise schedule is *not* static. It shifts based on the sequence length (total tokens).
    - **Formula**: The mean shift is linearly interpolated between `min_shift` (0.95 at 1024 tokens) and `max_shift` (2.05 at 4096 tokens). This allows the model to adapt its noise distribution for higher-resolution training.
- **Validation** (`validation_sampler.py`):
    - **Mini-Inference Engine**: The trainer includes a full (albeit simplified) inference pipeline for validation.
    - **Tiled Decoding**: Supports `TiledDecodingConfig` to split large generated latents into temporal/spatial tiles during validation decoding, essential for validating 4K/long-context generation on limited VRAM.

### 13.5 Data Processing Logic
- **Resolution Bucketing** (`process_videos.py`):
    - Standard noise application.
    - **Conditioning Tokens**: First-frame conditioning tokens are *not* noised. They are also explicitly **masked out of the loss** (`video_loss_mask = ~conditioning_mask`).
- **Video-to-Video (IC-LoRA)** (`video_to_video.py`):
    - **Concatenation**: Reference latents (clean) are concatenated with Target latents (noisy) along the temporal dimension.
    - **Reference Masking**: The *entire* reference portion of the sequence has `loss_mask = False`. The model sees the reference but is never penalized for predicting it.
    - **Downscaling**: The system automatically infers reference downscaling factors from the first batch and enforces consistency.

### 13.6 Data Processing Logic
- **Resolution Bucketing** (`process_videos.py`):
    - Uses a specific **Lexicographic Sort** to assign videos to buckets:
        1.  **Minimize Aspect Ratio Diff**: (log-scale) Primary factor.
        2.  **Maximize Frame Count**: Prefer buckets that use more of the video's temporal information.
        3.  **Maximize Spatial Area**: Prefer higher resolutions.
- **Tiled Encoding**:
    - Hardcoded `latent_channels = 128` in `tiled_encode_video`.
    - Manually splits tensors, encodes chips, and blends them using the `tiling` logic described above.

## 14. Glossary

- **DiT**: Diffusion Transformer.
- **RoPE**: Rotary Positional Embeddings (3D for video, 1D for audio).
- **STG**: Spatio-Temporal Guidance (perturbation-based guidance).
- **VAE**: Variational Autoencoder.
- **AdaLN**: Adaptive Layer Normalization.
- **IC-LoRA**: In-Context LoRA (Video-to-Video). Uses a reference video as a condition to guide the generation process.
- **Thinking Token**: A special learnable token in the Gemma text encoder connector that bridges text embeddings with the video/audio latent space.

---

## 15. Milimo System Integration

### 15.1 Pipeline Orchestration (`backend/tasks/video.py`)
Milimo does not use the `ltx-pipelines` directly via CLI. Instead, it wraps them in a stateful `generate_video_task` managed by `ModelManager`.

**Pipeline Selection Logic**:
1.  **Video Conditioning**: If the timeline contains video clips, `ICLoraPipeline` (In-Context LoRA) is selected.
2.  **Keyframe Interpolation**: If start (0) and end (N) keyframes are present, `KeyframeInterpolationPipeline` is selected.
3.  **Standard T2V/I2V**: Otherwise, `TI2VidTwoStagesPipeline` is used.
4.  **Single Frame**: If `num_frames=1`, the task delegates to **Flux 2** (`flux_inpainter`) for higher quality static image generation.

### 15.2 Quantum Alignment (Latent Handoff)
Implemented in `backend/tasks/chained.py`, this is Milimo's solution for infinite video extension ("Smart Continue").
*   **Mechanism**: Instead of using pixel-based Image-to-Video (which degrades quality over time), Milimo passes the **Last Latent Tensor** from Chunk N as the input conditioning for Chunk N+1.
*   **Logic**:
    1.  Extracts the last $K$ latent slices from the previous generation.
    2.  Passes them to the pipeline (custom `previous_latent_tensor` argument).
    3.  **Trimming**: The model regenerates these $K$ frames as "context". The system trims the first $K \times 8$ pixel frames from the output video to ensure perfect motion continuity without "skip-back" artifacts.
*   **Prompt Continuity**: Uses **Gemma 3** as a "Virtual Director" to rewrite the prompt for the next chunk based on the previous context, ensuring narrative consistency.

### 15.3 Hardware Optimizations (MPS)
*   **Tiling**: Milimo strictly enforces `TilingConfig.default()` (256px spatial tiles, 32 frame temporal tiles) on Apple Silicon.
*   **BFloat16**: Unlike Flux (which needs float32 hacks), LTX-2 runs fully in `bfloat16` on MPS.
*   **Cleanup**: Aggressive garbage collection (`gc.collect()` + `empty_cache()`) is performed between every chunk in a chained generation.
