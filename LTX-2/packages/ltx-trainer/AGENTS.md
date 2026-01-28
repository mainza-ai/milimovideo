# AGENTS.md

This file provides guidance to AI coding assistants (Claude, Cursor, etc.) when working with code in this repository.

## Project Overview

**LTX-2 Trainer** is a training toolkit for fine-tuning the Lightricks LTX-2 audio-video generation model. It supports:

- **LoRA training** - Efficient fine-tuning with adapters
- **Full fine-tuning** - Complete model training
- **Audio-video training** - Joint audio and video generation
- **IC-LoRA training** - In-context control adapters for video-to-video transformations

**Key Dependencies:**

- **[`ltx-core`](../ltx-core/)** - Core model implementations (transformer, VAE, text encoder)
- **[`ltx-pipelines`](../ltx-pipelines/)** - Inference pipeline components

> **Important:** This trainer only supports **LTX-2** (the audio-video model). The older LTXV models are not supported.

## Architecture Overview

### Package Structure

```
packages/ltx-trainer/
├── src/ltx_trainer/           # Main training module
│   ├── config.py              # Pydantic configuration models
│   ├── trainer.py             # Main training orchestration with Accelerate
│   ├── model_loader.py        # Model loading using ltx-core
│   ├── validation_sampler.py  # Inference for validation samples
│   ├── datasets.py            # PrecomputedDataset for latent-based training
│   ├── training_strategies/   # Strategy pattern for different training modes
│   │   ├── __init__.py        # Factory function: get_training_strategy()
│   │   ├── base_strategy.py   # TrainingStrategy ABC, ModelInputs, TrainingStrategyConfigBase
│   │   ├── text_to_video.py   # TextToVideoStrategy, TextToVideoConfig
│   │   └── video_to_video.py  # VideoToVideoStrategy, VideoToVideoConfig
│   ├── timestep_samplers.py   # Flow matching timestep sampling
│   ├── captioning.py          # Video captioning utilities
│   ├── video_utils.py         # Video processing utilities
│   └── hf_hub_utils.py        # HuggingFace Hub integration
├── scripts/                   # User-facing CLI tools
│   ├── train.py               # Main training script
│   ├── process_dataset.py     # Dataset preprocessing
│   ├── process_videos.py      # Video latent encoding
│   ├── process_captions.py    # Text embedding computation
│   ├── caption_videos.py      # Automatic video captioning
│   ├── decode_latents.py      # Latent decoding for debugging
│   ├── inference.py           # Inference with trained models
│   ├── compute_reference.py   # Generate IC-LoRA reference videos
│   └── split_scenes.py        # Scene detection and splitting
├── configs/                   # Example training configurations
│   ├── ltx2_av_lora.yaml      # Audio-video LoRA training
│   ├── ltx2_v2v_ic_lora.yaml  # IC-LoRA video-to-video
│   └── accelerate/            # Accelerate configs for distributed training
└── docs/                      # Documentation
```

### Key Architectural Patterns

**Model Loading:**

- `ltx_trainer.model_loader` provides component loaders using `ltx-core`
- Individual loaders: `load_transformer()`, `load_video_vae_encoder()`, `load_video_vae_decoder()`, `load_text_encoder()`, etc.
- Combined loader: `load_model()` returns `LtxModelComponents` dataclass
- Uses `SingleGPUModelBuilder` from ltx-core internally

**Training Flow:**

1. Configuration loaded via Pydantic models in `config.py`
2. `Trainer` class orchestrates the training loop
3. Training strategies (`TextToVideoStrategy`, `VideoToVideoStrategy`) prepare inputs and compute loss
4. Accelerate handles distributed training and device placement
5. Data flows as precomputed latents through `PrecomputedDataset`

**Model Interface (Modality-based):**

```python
from ltx_core.model.transformer.modality import Modality

# Create modality objects for video and audio
video = Modality(
    enabled=True,
    latent=video_latents,      # [B, seq_len, 128]
    timesteps=video_timesteps,  # [B, seq_len] per-token
    positions=video_positions,  # [B, 3, seq_len, 2]
    context=video_embeds,
    context_mask=None,
)
audio = Modality(
    enabled=True,
    latent=audio_latents,
    timesteps=audio_timesteps,
    positions=audio_positions,  # [B, 1, seq_len, 2]
    context=audio_embeds,
    context_mask=None,
)

# Forward pass returns predictions for both modalities
video_pred, audio_pred = model(video=video, audio=audio, perturbations=None)
```

> **Note:** `Modality` is immutable (frozen dataclass). Use `dataclasses.replace()` to modify.

**Configuration System:**

- All config in `src/ltx_trainer/config.py`
- Main class: `LtxTrainerConfig`
- Training strategy configs: `TextToVideoConfig`, `VideoToVideoConfig`
- Uses Pydantic field validators and model validators
- Config files in `configs/` directory

## Development Commands

### Setup and Installation

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
```

### Code Quality

```bash
# Run ruff linting and formatting
uv run ruff check .
uv run ruff format .

# Run pre-commit checks
uv run pre-commit run --all-files
```

### Running Tests

```bash
cd packages/ltx-trainer
uv run pytest
```

### Running Training

```bash
# Single GPU
uv run python scripts/train.py configs/ltx2_av_lora.yaml

# Multi-GPU with Accelerate
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml
```

## Code Standards

### Type Hints

- **Always use type hints** for all function arguments and return values
- Use Python 3.10+ syntax: `list[str]` not `List[str]`, `str | Path` not `Union[str, Path]`
- Use `pathlib.Path` for file operations

### Class Methods

- Mark methods as `@staticmethod` if they don't access instance or class state
- Use `@classmethod` for alternative constructors

### AI/ML Specific

- Use `@torch.inference_mode()` for inference (prefer over `@torch.no_grad()`)
- Use `accelerator.device` for distributed compatibility
- Support mixed precision (`bfloat16` via dtype parameters)
- Use gradient checkpointing for memory-intensive training

### Logging

- Use `from ltx_trainer import logger` for all messages
- Avoid print statements in production code

## Important Files & Modules

### Configuration (CRITICAL)

**`src/ltx_trainer/config.py`** - Master config definitions

Key classes:
- `LtxTrainerConfig` - Main configuration container
- `ModelConfig` - Model paths and training mode
- `TrainingStrategyConfig` - Union of `TextToVideoConfig` | `VideoToVideoConfig`
- `LoraConfig` - LoRA hyperparameters
- `OptimizationConfig` - Learning rate, batch size, etc.
- `ValidationConfig` - Validation settings
- `WandbConfig` - W&B logging settings

**⚠️ When modifying config.py:**
1. Update ALL config files in `configs/`
2. Update `docs/configuration-reference.md`
3. Test that all configs remain valid

### Training Core

**`src/ltx_trainer/trainer.py`** - Main training loop

- Implements distributed training with Accelerate
- Handles mixed precision, gradient accumulation, checkpointing
- Uses training strategies for mode-specific logic

**`src/ltx_trainer/training_strategies/`** - Strategy pattern

- `base_strategy.py`: `TrainingStrategy` ABC, `ModelInputs` dataclass
- `text_to_video.py`: Standard text-to-video (with optional audio)
- `video_to_video.py`: IC-LoRA video-to-video transformations

Key methods each strategy implements:
- `get_data_sources()` - Required data directories
- `prepare_training_inputs()` - Convert batch to `ModelInputs`
- `compute_loss()` - Calculate training loss
- `requires_audio` property - Whether audio components needed

**`src/ltx_trainer/model_loader.py`** - Model loading

Component loaders:
- `load_transformer()` → `LTXModel`
- `load_video_vae_encoder()` → `VideoVAEEncoder`
- `load_video_vae_decoder()` → `VideoVAEDecoder`
- `load_audio_vae_decoder()` → `AudioVAEDecoder`
- `load_vocoder()` → `Vocoder`
- `load_text_encoder()` → `AVGemmaTextEncoderModel`
- `load_model()` → `LtxModelComponents` (convenience wrapper)

**`src/ltx_trainer/validation_sampler.py`** - Inference for validation

Uses ltx-core components for denoising:
- `LTX2Scheduler` for sigma scheduling
- `EulerDiffusionStep` for diffusion steps
- `CFGGuider` for classifier-free guidance

### Data

**`src/ltx_trainer/datasets.py`** - Dataset handling

- `PrecomputedDataset` loads pre-computed VAE latents
- Supports video latents, audio latents, text embeddings, reference latents

## Common Development Tasks

### Adding a New Configuration Parameter

1. Add field to appropriate config class in `src/ltx_trainer/config.py`
2. Add validator if needed
3. Update ALL config files in `configs/`
4. Update `docs/configuration-reference.md`

### Implementing a New Training Strategy

1. Create new file in `src/ltx_trainer/training_strategies/`
2. Create config class inheriting `TrainingStrategyConfigBase`
3. Create strategy class inheriting `TrainingStrategy`
4. Implement: `get_data_sources()`, `prepare_training_inputs()`, `compute_loss()`
5. Add to `__init__.py`: import, add to `TrainingStrategyConfig` union, update factory
6. Add discriminator tag to config.py's `TrainingStrategyConfig`
7. Create example config file in `configs/`

### Working with Modalities

```python
from dataclasses import replace
from ltx_core.model.transformer.modality import Modality

# Create modality
video = Modality(
    enabled=True,
    latent=latents,
    timesteps=timesteps,
    positions=positions,
    context=context,
    context_mask=None,
)

# Update (immutable - must use replace)
video = replace(video, latent=new_latent, timesteps=new_timesteps)

# Disable a modality
audio = replace(audio, enabled=False)
```

## Debugging Tips

**Training Issues:**

- Check logs first (rich logger provides context)
- GPU memory: Look for OOM errors, enable `enable_gradient_checkpointing: true`
- Distributed training: Check `accelerator.state` and device placement

**Model Loading:**

- Ensure `model_path` points to a local `.safetensors` file
- Ensure `text_encoder_path` points to a Gemma model directory
- URLs are NOT supported for model paths

**Configuration:**

- Validation errors: Check validators in `config.py`
- Unknown fields: Config uses `extra="forbid"` - all fields must be defined
- Strategy validation: IC-LoRA requires `reference_videos` in validation config

## Key Constraints

### LTX-2 Frame Requirements

Frames must satisfy `frames % 8 == 1`:
- ✅ Valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- ❌ Invalid: 24, 32, 48, 64, 100

### Resolution Requirements

Width and height must be divisible by 32.

### Model Paths

- Must be local paths (URLs not supported)
- `model_path`: Path to `.safetensors` checkpoint
- `text_encoder_path`: Path to Gemma model directory

### Platform Requirements

- Linux required (uses `triton` which is Linux-only)
- CUDA GPU with 24GB+ VRAM recommended

## Reference: ltx-core Key Components

```
packages/ltx-core/src/ltx_core/
├── model/
│   ├── transformer/
│   │   ├── model.py              # LTXModel
│   │   ├── modality.py           # Modality dataclass
│   │   └── transformer.py        # BasicAVTransformerBlock
│   ├── video_vae/
│   │   └── video_vae.py          # Encoder, Decoder
│   ├── audio_vae/
│   │   ├── audio_vae.py          # Decoder
│   │   └── vocoder.py            # Vocoder
│   └── clip/gemma/
│       └── encoders/av_encoder.py  # AVGemmaTextEncoderModel
├── pipeline/
│   ├── components/
│   │   ├── schedulers.py         # LTX2Scheduler
│   │   ├── diffusion_steps.py    # EulerDiffusionStep
│   │   ├── guiders.py            # CFGGuider
│   │   └── patchifiers.py        # VideoLatentPatchifier, AudioPatchifier
│   └── conditioning/             # VideoLatentTools, AudioLatentTools
└── loader/
    ├── single_gpu_model_builder.py  # SingleGPUModelBuilder
    └── sd_ops.py                    # Key remapping (SDOps)
```
