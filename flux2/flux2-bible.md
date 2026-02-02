# FLUX.2 Repository Documentation

## 1. Executive Summary

FLUX.2 is a state-of-the-art image generation and editing system developed by Black Forest Labs. This repository contains the reference inference implementation for the FLUX.2 family of models, which includes the **[klein]** (distilled, low-latency, 4B/9B parameters) and **[dev]** (32B parameters) variants.

The system relies on a **Rectified Flow Transformer** architecture. It supports:
- **Text-to-Image (T2I)** generation.
- **Image-to-Image (I2I)** editing with single or multiple reference images.
- **Prompt Upsampling** via local or remote Large Language Models (LLMs).
- **Invisible Watermarking** for provenance (implied by codebase references).

This documentation serves as the technical reference for maintaining, extending, and integrating FLUX.2.

## 2. System Architecture Overview

### Architectural Style
The system follows a modular, inference-focused pipeline architecture:
1.  **Conditioning Phase**: Text prompts are encoded into embeddings; reference images are encoded into latent representations.
2.  **Generative Phase**: A Transformer-based flow model denoises random latent noise conditioned on the inputs.
3.  **Decoding Phase**: The resulting latents are decoded back into pixel space via an Autoencoder.

### Major Subsystems
-   **Flow Model (`Flux2`)**: The core backbone handling the generative process. It uses a mix of "Double Stream" blocks (processing text and image tokens jointly) and "Single Stream" blocks.
-   **Text Encoder**: Wraps LLMs (Mistral or Qwen) to generate rich text embeddings.
-   **Autoencoder (AE)**: Compresses images into latent space and reconstructs them.
-   **Sampling Orchestrator**: Manages the denoising scheduler and loop.

### Runtime Environment
-   **Languages**: Python 3.10+
-   **Frameworks**: PyTorch, SafeTensors, HuggingFace Transformers.
-   **Hardware assumptions**: CUDA-capable GPUs. High VRAM requirements for the [dev] model (H100 class), manageable on consumer hardware for [klein] variants.

## 3. Repository Structure

```
├── assets/                 # Sample images and visual assets
├── docs/                   # supplementary documentation
├── model_cards/            # Model definitions and metadata
├── model_licenses/         # Legal licenses
├── scripts/
│   └── cli.py              # Main entry point for interactive inference
├── src/
│   └── flux2/
│       ├── autoencoder.py  # VAE implementation
│       ├── model.py        # Core Flux2 Transformer foundation
│       ├── openrouter_api_client.py # Client for remote prompt upsampling
│       ├── sampling.py     # Diffusion/Flow matching scheduling and loop
│       ├── system_messages.py # System prompts for the upsampling/safety LLMs
│       ├── text_encoder.py # Wrappers for Mistral/Qwen text encoders
│       ├── util.py         # Model loading and configuration utilities
│       └── watermark.py    # Watermarking logic (referenced but optionally not fully used in all paths)
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # User-facing introduction
```

## 4. Core Concepts & Design Philosophy

-   **Flow Matching**: The generation process is modeled as a straight path interpolation between noise and data. This is evident in the `denoise` function: `img + (t_prev - t_curr) * pred`.
-   **Joint Attention**: The "Double Stream" blocks in the transformer allow bidirectional attention between text and image tokens, enabling high adherence to prompts.
-   **Rotary Positional Embeddings (RoPE)**: Used for encoding spatial information in the transformer.
-   **Distillation**: The `[klein]` models utilize guidance distillation (adversarial or similar) allowing for very few inference steps (e.g., 4 steps), whereas `[dev]` and base models use standard scheduling (~50 steps).

## 5. Detailed Module & File Documentation

### 5.1 `src/flux2`

#### `model.py`
**Purpose**: Defines the neural network architecture for the generative model.
**Key Classes**:
-   `Flux2`: The main `nn.Module`. Initializes embedding layers and the stack of transformer blocks.
-   `DoubleStreamBlock`: **Joint Attention Mechanism**.
    -   Text and Image tokens are processed in parallel streams.
    -   **Full Bidirectional Interaction**: Q, K, V from both streams are concatenated (`q = cat(txt_q, img_q)`, etc.) allowing text to attend to image and vice versa within the same self-attention operation.
    -   Uses separate modulation signatures (`img_mod`, `txt_mod`) for adaptivity.
-   `SingleStreamBlock`: Concatenates image and text tokens into a single sequence for the final processing stages. Uses a standard DiT block structure.
-   `EmbedND`: Implements N-dimensional Rotary Positional Embeddings (RoPE).
    -   **Critical Detail**: Uses a Base Theta of **2000** (via `Flux2Params`), not the standard 10000 often used in LLMs.
    -   Encodes Position IDs across 3 dimensions (Time, Height, Width) or more, concatenated into a single embedding.

**Model Variants & Parameters**:
| Model Variant | Hidden Size | Depth (D/S) | MLP Ratio | Default Steps | Default Guidance | Distilled? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[dev]** | 6144 | 8 / 48 | 4.0 | 50 | 4.0 | Yes |
| **[klein] 9B** | 4096 | 8 / 24 | 3.0 | 4 | 1.0 | Yes |
| **[klein] 4B** | 3072 | 5 / 20 | 3.0 | 4 | 1.0 | Yes |
| **[klein] Base 9B**| 4096 | 8 / 24 | 3.0 | 50 | 4.0 | No |
| **[klein] Base 4B**| 3072 | 5 / 20 | 3.0 | 50 | 4.0 | No |

#### `sampling.py`
**Purpose**: Implements the diffusion/flow generation loop.
**Key Algorithms**:
-   **Resolution-Dependent Scheduling (`get_schedule`)**:
    -   Calculates a `mu` shift based on sequence length (`image_seq_len`).
    -   **Magic Numbers**:
        -   Uses linear interpolation between `m_10` (seq_len ~10) and `m_200` (seq_len ~200).
        -   **Breakpoint**: If `image_seq_len > 4300`, it switches to a direct linear formula `mu = a2 * len + b2`.
    -   Formula: `math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** 1.0)`.
-   **Reference Image Encoding**:
    -   `encode_image_refs` places reference images at specific time offsets (`scale + scale * t`, where scale=10) to separate them from the target generation in the temporal position embedding space.
    -   Applies strict pixel limits (~4MP for single ref) to manage memory.
-   **ID Scattering**: `scatter_ids` and `compress_time` manage the mapping of flattened tokens back to (T, H, W) spatial positions.

#### `autoencoder.py`
**Purpose**: Implements the Variational Autoencoder (VAE) for compressing images.
**Key Features**:
-   **Latent Normalization**: The `AutoEncoder` class includes a specific `BatchNorm2d` layer (`self.bn`) used to normalize latents to zero-mean/unit-variance before they enter the flow model. This acts as a fixed pre-processing step separate from the core Encoder/Decoder weights.
-   **Architecture**: Standard ResNet-based Encoder/Decoder with `ch_mult=[1, 2, 4, 4]`, resulting in a high compression factor.
#### `system_messages.py`
**Purpose**: Stores the system prompts that govern the behavior of auxiliary LLMs (upsamplers and safety filters).
**Key Components**:
-   `SYSTEM_MESSAGE_UPSAMPLING_T2I`: Instructs the LLM to rewrite prompts with "concrete visual specifics" (lighting, texture) while quoting all text elements.
-   `SYSTEM_MESSAGE_UPSAMPLING_I2I`: Instructs the LLM to convert editing requests into a single, concise positive instruction (e.g., "don't change X" -> "keep X").
-   `PROMPT_IMAGE_INTEGRITY`: Criteria for determining if an image contains copyrighted material or public figures.

#### `watermark.py`
**Purpose**: Implements invisible watermarking to stamp generated images.
**Mechanism**:
-   Uses the `imwatermark` library (based on DWT/DCT transform).
-   Embeds a **fixed 48-bit message** (`0b00101010...`) derived from a random initialization.
-   Operates on the image in BGR format (converting from RGB).

#### `text_encoder.py`
**Purpose**: Wraps external LLMs to provide text embeddings for conditioning and performs safety checks.
**Key Classes**:
-   `Mistral3SmallEmbedder`: Wraps `Mistral3ForConditionalGeneration`. Used for FLUX.2 [dev].
    -   **Prompt Formatting**: Explicitly cleans `[IMG]` tokens from prompts to prevent processor validation errors.
    -   Constructs a structured conversation: `[{"role": "system", ...}, {"role": "user", "content": [{"type": "text", ...}]}]`.
-   `Qwen3Embedder`: Wraps `Qwen/Qwen3`. Used for FLUX.2 [klein].
-   `upsample_prompt`: Uses the LLM capabilities (locally or via OpenRouter) to expand simple user prompts into detailed descriptions.
-   **Safety Mechanism**: Contains `test_image` and `test_txt`.
    -   **Image Safety**: Uses `Falconsai/nsfw_image_detection` (CLIP-based) + LLM-based analysis for copyright/public figures.
    -   **Text Safety**: LLM-based analysis of the prompt against `PROMPT_TEXT_INTEGRITY` criteria.

#### `util.py`
**Purpose**: Utilities for model loading and configuration.
**Key Components**:
-   `FLUX2_MODEL_INFO`: A dictionary registry containing configuration (repo IDs, filenames, parameter classes) for all supported model variants (`klein-4b`, `dev`, etc.).

#### `scripts/cli.py`
**Purpose**: The primary interactive interface for the user.
**Responsibilities**:
-   Parsing user input for configuration (width, height, steps).
-   **Argument Parsing Quirks**: Uses `shlex` to parse key-value pairs (`key=value`). Lists like `input_images` can be comma command/space separated but must be quoted if containing spaces.
-   **Seeding**: If `seed` is not provided (None), it generates a random `int64` seed (`randrange(2**31)`).
-   **Resolution Matching**: Can optionally match output resolution to an input image (`match_image_size=<index>`).

## 6. Data Flow & Control Flow

1.  **Initialization**:
    -   User runs `scripts/cli.py`.
    -   `main()` selects the model variant (e.g., `flux.2-klein-4b`) and loads separate components: `TextEncoder`, `Flux2` (Flow Model), `AutoEncoder`.

2.  **Generation Loop**:
    -   **Input**: User types a prompt.
    -   **Prompt Enhancement (Optional)**: If upsampling is enabled, `TextEncoder` (or OpenRouter API) rewrites the prompt.
    -   **Conditioning**: `TextEncoder` converts text -> embeddings (`ctx`).
    -   **Latent Initialization**: Random noise (`randn`) is generated on GPU.
    -   **Denoising**: `sampling.denoise()` iterates `num_steps` times.
        -   In each step, `Flux2` model predicts the update direction.
        -   Latents are updated: $x_{t-1} = x_t + (t_{prev} - t_{curr}) * v_{pred}$.
    -   **Decoding**: The final latent $x_0$ is passed to `AutoEncoder.decode()`.
    -   **Output**: The resulting tensor is converted to a PIL Image and saved.

## 7. Configuration & Environment Variables

### Runtime Environment
-   **Dependencies**: Requires specific versions as per `pyproject.toml`: `torch==2.8.0`, `torchvision==0.23.0`, `transformers==4.56.1`.
-   **Configuration**:
    -   `FLUX2_MODEL_PATH`, `AE_MODEL_PATH`: Environment variables overrides for weight paths.
    -   `OPENROUTER_API_KEY`: API key for remote prompt upsampling.

### CLI Configuration
The `Config` dataclass in `cli.py` defines runtime parameters:
-   `width`, `height`: Output resolution (must affect latent dimensions).
-   `num_steps`: Number of sampling steps (4 for distilled, ~50 for base).
-   `guidance`: float scale (usually 1.0 for distilled, higher for base).

## 8. External Integrations

-   **HuggingFace Hub**: Used for automatic weight downloading.
-   **OpenRouter**: Optional external API integration for prompt enhancement/upsampling. The system is optimized for `mistralai/pixtral-large-2411`.
-   **Invisible Watermark**: Uses `imwatermark` (DWT/DCT based) to embed a static 48-bit signature.

## 9. Extension & Customization Guide

-   **Adding New Models**: Update `FLUX2_MODEL_INFO` in `src/flux2/util.py` with the new model's HuggingFace ID and parameter dataclass.
-   **Customizing Sampling**: Modify `src/flux2/sampling.py`. If implementing a completely different sampler (e.g., Euler, DPM++), add the function there and update `cli.py` to call it.
-   **New Text Encoders**: Subclass `nn.Module` in `src/flux2/text_encoder.py` and implement the `forward` method to return embeddings in the expected shape.
-   **Modifying Safety**: The safety filter logic is in `text_encoder.py`. You can adjust `NSFW_THRESHOLD` (default 0.85) or modify the system prompts in `system_messages.py` to change the strictness of copyright/public figure detection.

## 10. Debugging & Maintenance Guide

-   **VRAM Issues**: The code handles CPU offloading (`cpu_offloading=True` in `main`). If OOM occurs, ensure offloading is enabled or switch to quantized weights (not natively supported in this repo, requires `diffusers`).
-   **Shape Mismatches**: The model enforces strict division constraints on hidden sizes vs heads. `ValueError` in `model.py` usually indicates a configuration mismatch in `Flux2Params`.
-   **Upsampling Errors**: If OpenRouter fails, the system automatically falls back to the original prompt. Check `OPENROUTER_API_KEY` if upsampling is silently ignored.

## 11. Known Risks, Limitations & Technical Debt

-   **Resolution constraints**: `Flux2Params` assumes specific divisibility. Arbitrary resolutions might fail if not aligned with patch sizes.
-   **Hardcoded Paths**: Some defaults in `util.py` point to specific HF repos that might change.
-   **Sync vs Async**: The code is synchronous. Real-time applications might need to wrap the `main` generation logic in an async API.
-   **Safety**:
    -   The system enforces safety checks by loading the `flux.2-dev` text encoder (Mistral) even when running `klein` models (`mod_and_upsampling_model`).
    -   Input prompts are checked (`test_txt`) and Input/Output images are checked (`test_image`).
    -   **Note**: Safety filters use `falconsai/nsfw_image_detection` and LLM-based verification.
-   **Watermarking**: The `watermark.py` module exists but is **commented out** in `cli.py` by default (`# x = embed_watermark(x)`). To enable watermarking, uncomment these lines.

## 12. Glossary

- **DiT (Diffusion Transformer)**: A class of diffusion models that replace the traditional U-Net backbone with a Transformer architecture, allowing for better scaling and scalability.
- **Flow Matching**: A generative modeling paradigm where the model learns to matched a specific vector field (flow) that transforms a simple prior distribution (noise) to the data distribution.
- **RoPE (Rotary Positional Embeddings)**: A method for encoding positional information in Transformers by rotating query and key vectors in a high-dimensional space.
- **Latent Space**: A compressed representation of the image data, produced by the Autoencoder, where the diffusion process actually takes place (Latent Diffusion).
- **Distillation**: A process of training a smaller or faster student model (e.g., [klein]) to mimick the behavior of a larger or slower teacher model (e.g., [dev]), often reducing the number of required inference steps.
- **CFG (Classifier-Free Guidance)**: A technique to improve sample quality and prompt adherence by interpolating between conditional (prompted) and unconditional (empty prompt) noise predictions.
