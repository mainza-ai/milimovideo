# LTX-2 Repository Documentation

## 1. Executive Summary

LTX-2 (Lightricks Text-to-eXperience 2) is a **joint video and audio generation system** that generates synchronized audiovisual content from text prompts. It uses a **19-billion-parameter Dual-Stream Transformer** that processes video and audio tokens in parallel within a unified latent space.

The system supports:
- **Text-to-Video (T2V)** with optional audio generation.
- **Image-to-Video (I2V)** using input images as conditioning frames.
- **IC-LoRA** integration for subject consistency (concept injection).
- **Keyframe Interpolation** between start/end frames.
- **Temporal and Spatial Upsampling** for resolution enhancement.
- **Chained (Autoregressive) Generation** for videos exceeding the 121-frame context window.

The Milimo system integrates LTX-2 as its primary video generation engine through the `ModelManager` singleton and custom task dispatchers.

## 2. System Architecture Overview

### Architectural Style
LTX-2 follows a multi-stage generative pipeline:

1.  **Encoding Phase**: Text → Gemma 3 embeddings; Reference images → VAE latents.
2.  **Latent Diffusion Phase**: A Dual-Stream Transformer denoises latent noise into video + audio latent representations.
3.  **Decoding Phase**: Latents → pixels/audio waveforms via separate VAE decoders.
4.  **Upsampling Phase (Optional)**: Spatial (×2) and/or Temporal (×2) upsamplers enhance final output.

### Major Subsystems
- **Dual-Stream Transformer (19B)**: The backbone of the system. Processes video and audio tokens in separate FFN streams but with shared attention, ensuring synchronization.
- **Gemma 3 (Text Encoder)**: Google's Gemma language model variant providing rich text representations.
- **Video VAE**: Handles compression of video frames into and out of the latent space.
- **Audio VAE**: Handles audio waveform compression.
- **Distilled LoRA (384)**: A rank-384 LoRA that enables few-step generation (~8 steps) from the distilled checkpoint.
- **Spatial Upsampler (×2)**: Post-generation resolution doubler.
- **Temporal Upsampler (×2)**: Doubles framerate of generated video.

### Runtime Environment
- **Languages**: Python 3.10+
- **Frameworks**: PyTorch, SafeTensors, HuggingFace Transformers.
- **Hardware**: CUDA GPUs (H100 class for optimal performance). Apple Silicon MPS supported with dtype workarounds.

## 3. Repository Structure

```
LTX-2/
├── packages/
│   ├── ltx-core/             # Core model and loader definitions
│   │   └── src/ltx_core/
│   │       └── loader.py     # SafeTensor loading, LoRA merge, registry
│   ├── ltx-pipelines/        # Generator pipeline implementations
│   │   └── src/ltx_pipelines/
│   │       ├── ti2vid_two_stages.py  # Text/Image → Video (primary)
│   │       ├── ic_lora.py            # IC-LoRA concept injection
│   │       └── keyframe_interpolation.py # Start→End interpolation
│   └── ltx-training/        # Training scripts (not used at inference)
├── media/                    # Sample outputs
├── models/                   # Weight storage
│   ├── checkpoints/          # *.safetensors main model weights
│   ├── upscalers/            # Spatial/Temporal upsampler weights
│   └── text_encoders/gemma3/ # Gemma 3 text encoder weights
└── README.md
```

## 4. Core Concepts & Design Philosophy

- **Dual-Stream Architecture**: Unlike single-stream diffusion models, LTX-2 maintains separate FFN pathways for video and audio tokens but uses **shared self-attention** across both streams. This ensures audiovisual temporal alignment without explicit synchronization.
- **Flow Matching + Distillation**: Uses rectified flow matching for training, then a rank-384 LoRA distillation to reduce inference from ~50 steps to ~8 steps.
- **Two-Stage Pipeline**: The primary `TI2VidTwoStagesPipeline` runs:
  - **Stage 1 (Generation)**: Full Transformer denoising at low latent resolution.
  - **Stage 2 (Upsampling)**: Spatial (×2) and optionally Temporal (×2) upsampling on the decoded output.
- **Image Conditioning**: Input images are encoded into the video VAE latent space and injected as temporal conditioning points. The model attends to these latents during denoising.

## 5. Pipeline Types

### 5.1 `TI2VidTwoStagesPipeline`
**Purpose**: The workhorse pipeline for text-to-video and image-conditioned video generation.

**Parameters**:
- `prompt`: Text description.
- `width`, `height`: Output resolution (before upsampling).
- `num_frames`: Target frame count (max 121 per chunk).
- `num_images_cond`: List of `(path, frame_idx, strength)` tuples for conditioning.
- `seed`: Reproducibility seed.
- `guidance_scale`: CFG-like scale (typically 1.0 for distilled).

### 5.2 `ICLoraPipeline`
**Purpose**: Concept injection via IC-LoRA. Maintains identity/style consistency of subjects across generations.

**Key difference from TI2Vid**: Takes dedicated `loras` argument pointing to IC-LoRA weight files. Does not use the distilled LoRA.

### 5.3 `KeyframeInterpolationPipeline`
**Purpose**: Given start and end frame images, generates a smooth video transition between them.

**Parameters**: Similar to TI2Vid but requires exactly 2 conditioning images (frame 0 → start, frame N → end).

## 6. Model & Checkpoint Variants

| Checkpoint | Size | Precision | MPS Compatible |
|---|---|---|---|
| `ltx-2-19b-distilled.safetensors` | Full BF16 | bfloat16/float32 | ✅ (with float32 cast) |
| `ltx-2-19b-distilled-fp8.safetensors` | Quantized FP8 | float8_e4m3fn | ❌ (requires CUDA) |
| `ltx-2-19b-distilled-lora-384.safetensors` | LoRA rank-384 | — | ✅ |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | Spatial ×2 | — | ✅ |
| `ltx-2-temporal-upscaler-x2-1.0.safetensors` | Temporal ×2 | — | ✅ |

**Auto-Selection Logic** (in `ModelManager.get_model_paths()`):
1.  Priority: Full precision (`ltx-2-19b-distilled.safetensors`).
2.  Fallback: FP8 (`ltx-2-19b-distilled-fp8.safetensors`).

## 7. Data Flow

### Standard Generation (`TI2VidTwoStagesPipeline`)
1.  **Text Encoding**: `prompt` → Gemma 3 → text embeddings.
2.  **Image Conditioning**: Input images → Video VAE encode → latents → injected at specified frame indices.
3.  **Noise Initialization**: Random latent of shape `(B, C, T, H, W)`.
4.  **Stage 1 (Denoising)**: 8 steps (distilled) through Dual-Stream Transformer.
5.  **Stage 1 Decode**: Latents → Video VAE decode → pixel frames.
6.  **Stage 2 (Spatial Upsample)**: ×2 spatial resolution enhancement.
7.  **Stage 2 (Temporal Upsample, optional)**: ×2 framerate doubling.
8.  **Output**: Video file (MP4 via `moviepy` or `ffmpeg`).

---

## 8. Milimo System Integration

### 8.1 Integration Architecture

Milimo integrates LTX-2 through three layers:

```
┌────────────────────────────────────────────────┐
│ Frontend (React)                                │
│   └─ Generate Button → POST /api/generate       │
├────────────────────────────────────────────────┤
│ server.py → BackgroundTasks                     │
│   └─ generate_video_task()                      │
├────────────────────────────────────────────────┤
│ tasks/video.py → Pipeline Selection             │
│   ├─ Standard Path → TI2Vid / ICLora / Keyframe │
│   └─ Chained Path → tasks/chained.py            │
├────────────────────────────────────────────────┤
│ model_engine.py → ModelManager (Singleton)      │
│   └─ Loads/caches pipeline instances            │
├────────────────────────────────────────────────┤
│ LTX-2 Pipelines (ltx_pipelines/*)              │
└────────────────────────────────────────────────┘
```

### 8.2 `ModelManager` (Singleton)

**File**: `backend/model_engine.py`  
**Instance**: `manager` (module-level singleton)

**Responsibilities**:
- **Pipeline Lifecycle**: Loads exactly one pipeline at a time. If the requested type matches the cached type, reuses it. Otherwise, unloads the old pipeline (with `gc.collect()` + `torch.mps.empty_cache()` or `torch.cuda.empty_cache()`) and loads the new one.
- **Checkpoint Selection**: Auto-detects best available checkpoint (full > FP8).
- **MPS Compatibility**: Disables FP8 transformer on MPS (`fp8transformer=False`).
- **MPS VAE Fix**: After loading, forces `pipeline.vae` to `float32` to prevent black video frames.

**Pipeline Construction Args**:
```python
TI2VidTwoStagesPipeline(
    checkpoint_path=paths["checkpoint_path"],
    distilled_lora=distilled_lora_objs,        # LoRA rank-384
    spatial_upsampler_path=paths["spatial_upsampler_path"],
    temporal_upsampler_path=paths["temporal_upsampler_path"],
    gemma_root=paths["gemma_root"],
    loras=loras,                                # User-specified LoRAs
    device=device,                              # "cuda" / "mps" / "cpu"
    fp8transformer=fp8                          # False on MPS
)
```

### 8.3 Pipeline Selection Logic (`tasks/video.py`)

**`generate_video_task()`** orchestrates the entire flow:

1.  **Auto-Detection** (when `pipeline_type == "advanced"` or `"auto"`):
    -   If timeline has **2+ conditioning images at different frames** → `"keyframe"`.
    -   If timeline has **IC-LoRA elements** → `"ic_lora"`.
    -   Default → `"ti2vid"`.

2.  **Chained Delegation**:
    -   If `pipeline_type == "ti2vid"` AND `num_frames > 121` → delegates to `generate_chained_video_task()`.

3.  **Single-Frame Shortcut**:
    -   If `num_frames == 1` → bypasses LTX-2 entirely, calls `flux_inpainter.generate_image()` from Flux 2.

### 8.4 Prompt Enhancement

**`generate_enhanced_prompt()`** (in `tasks/video.py`):
1.  Uses the **Gemma 3 text encoder** already loaded by the LTX-2 pipeline.
2.  If an input image is available, it captures a visual description from the image.
3.  Enhances the user prompt into a cinematically detailed description.
4.  Returns the enhanced prompt for use in generation.

### 8.5 Input Image Path Resolution

**Critical Logic** (in `generate_standard_video_task()`):
Timeline images arrive from the frontend as URLs (e.g., `http://localhost:5173/projects/abc/assets/my image.png`). The backend resolves them:

1.  **URL Detection**: If path starts with `http://` or `https://`:
    -   Parse with `urlparse` → extract `parsed.path`.
    -   **URL-decode** with `unquote(parsed.path)` (handles spaces `%20`, special chars).
2.  **Project Path Resolution**: If path starts with `/projects`:
    -   Strip prefix, join with `config.PROJECTS_DIR` to get absolute path.
3.  **Validation**: Check `os.path.exists()` on resolved path.

### 8.6 Chained (Autoregressive) Generation

**File**: `backend/tasks/chained.py`  
**Function**: `generate_chained_video_task()`

This is Milimo's "Smart Continue" feature for videos longer than 121 frames.

#### Chunk Calculation (via `StoryboardManager`):
```python
chunk_size = 121
overlap_frames = 24
effective_step = chunk_size - overlap_frames  # = 97
additional = ceil((total_frames - chunk_size) / effective_step)
total_chunks = 1 + additional
```

**Example**: 300 frames → `1 + ceil((300-121)/97)` = `1 + 2` = **3 chunks**.

#### Per-Chunk Loop:
For each chunk `i`:
1.  **Configuration Prep** (`StoryboardManager.prepare_next_chunk()`):
    -   Chunk 0: Uses the original prompt and any user-specified conditioning images.
    -   Chunk 1+: Extracts last frame of previous chunk (via `ffmpeg -sseof -0.1`) as conditioning at frame 0.
2.  **LTX-2 Generation**: Standard `TI2VidTwoStagesPipeline` call with 121 frames.
3.  **Cancellation Check**: After each chunk, checks `active_jobs[job_id]["cancelled"]`.
4.  **Progress Broadcasting**: SSE updates with per-chunk progress (`chunk {i}/{total}`).

#### Latent Handoff ("Quantum Alignment"):

The core mechanism for maintaining motion continuity between chunks:

1.  **Capture**: After each chunk's Stage 1 denoising, the full latent tensor is captured (`last_chunk_latent`).
2.  **Alignment Math**:
    -   `requested_pixel_overlap` = 24 frames.
    -   `latent_slice_count = ceil((requested_pixel_overlap - 1) / 8) + 1` — aligns pixel overlap to the latent grid (1 latent = 8 pixels in temporal dimension).
    -   `effective_overlap_pixels = (latent_slice_count - 1) × 8 + 1` — the actual overlap in pixel space after alignment.
    -   `frames_to_trim = effective_overlap_pixels` — exact number of decoded frames to trim from subsequent chunks.
3.  **Slicing**: `conditioning_latent_tensor = last_chunk_latent[:, :, start_slice:, :, :]` — the tail of the previous chunk's latent tensor.
4.  **Injection**: The sliced latent is passed to LTX-2 as a conditioning input, seeding the start of the next chunk with the end of the previous chunk.
5.  **Why "Quantum"**: The latent grid quantizes continuous pixel space. Overlap must be aligned to this grid to avoid discontinuities. Misaligned overlaps cause "frozen anchor" artifacts where the overlap region appears static.

#### Overlap Trimming & Video Stitching:
1.  For chunk `i > 0`: Trim exactly `frames_to_trim` frames from the **start** of the decoded video.
2.  Trimming is done via **ffmpeg**:
    ```
    ffmpeg -i chunk.mp4 -ss <trim_seconds> -c copy trimmed_chunk.mp4
    ```
    Where `trim_seconds = frames_to_trim / fps`.
3.  All trimmed chunks are concatenated using ffmpeg concat (file-list method or filter concat).

### 8.7 The `StoryboardManager`

**File**: `backend/storyboard/manager.py`

Orchestrates multi-chunk generation with narrative awareness.

**Dual Modes**:
1.  **Worker/Generation Mode**: Initialized with `(job_id, prompt, params, output_dir)`. Tracks chunk state during chained generation.
2.  **Server/Prep Mode**: Initialized with just `output_dir` for shot-based storyboard workflows.

**Key Methods**:
| Method | Purpose |
|---|---|
| `get_total_chunks()` | Calculates required chunks from `num_frames`, `chunk_size` (121), `overlap_frames` (24) |
| `prepare_next_chunk(chunk_idx, last_chunk_output)` | Returns `{prompt, images}` for next chunk. Extracts last frame via ffmpeg for conditioning |
| `commit_chunk(chunk_idx, path, prompt)` | Registers completed chunk in `StoryboardState.chunks` |
| `prepare_shot_generation(shot_id, session)` | For storyboard mode: resolves Shot→Scene→previous shot chain, enriches prompt via `element_manager` |
| `cleanup()` | Stub for artifact cleanup |

**Shot-based Prompt Enrichment** (`prepare_shot_generation`):
1.  Loads `Shot` from DB.
2.  Finds previous shot in same scene (`Shot.index - 1`).
3.  If previous shot is completed → extracts last frame as conditioning.
4.  Injects narrative context: `"Following the previous shot where {prev_shot.action}. {enriched_prompt}"`.
5.  Resolves project `Element` trigger words via `element_manager.inject_elements_into_prompt()`.

### 8.8 MPS Compatibility Summary

| Component | Issue | Fix |
|---|---|---|
| **Transformer** | `bfloat16` causes `MPSNDArrayMatrixMultiplication` error | Pipeline forces `float32` on MPS (via `fp8transformer=False` + loader casts) |
| **VAE** | `bfloat16` decode → black frames | `ModelManager` forces `pipeline.vae.to(dtype=torch.float32)` after load |
| **FP8 Checkpoint** | FP8 quantization unsupported on MPS | Auto-selects full-precision checkpoint instead |
| **Memory Management** | MPS fragmentation causes OOM mid-generation | `gc.collect()` + `torch.mps.empty_cache()` on pipeline swap |

### 8.9 Video Output & Persistence

After generation completes:
1.  Video saved to `projects/{project_id}/generated/{job_id}.mp4`.
2.  Thumbnail extracted (first frame) → `projects/{project_id}/thumbnails/{job_id}.jpg`.
3.  `Shot` record updated in DB: `video_url`, `thumbnail_url`, `actual_frames`, `status="completed"`.
4.  `Job` record updated: `status="completed"`, `output_path`, `thumbnail_path`.
5.  SSE broadcast: `{"type": "complete", "job_id": ..., "url": ..., "thumbnail_url": ...}`.

### 8.10 Weight Paths & Configuration

Configured in `config.py` and `model_engine.py`:

| Config | Path Pattern | Description |
|---|---|---|
| `config.LTX_DIR` | `<PROJECT_ROOT>/LTX-2` | Root of LTX-2 repo |
| Checkpoint | `LTX-2/models/checkpoints/ltx-2-19b-distilled.safetensors` | Main model weights |
| Distilled LoRA | `LTX-2/models/checkpoints/ltx-2-19b-distilled-lora-384.safetensors` | Few-step LoRA |
| Spatial Upsampler | `LTX-2/models/upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors` | ×2 spatial |
| Temporal Upsampler | `LTX-2/models/upscalers/ltx-2-temporal-upscaler-x2-1.0.safetensors` | ×2 temporal |
| Gemma 3 | `LTX-2/models/text_encoders/gemma3/` | Text encoder weights |
| IP-Adapter | `config.FLUX_IP_ADAPTER_PATH` | Flux IP-Adapter (cross-referenced from Flux config) |

## 9. Debugging & Maintenance Guide

- **Black Video Output**: Usually a VAE dtype issue on MPS. Verify `pipeline.vae.dtype == torch.float32`.
- **Frozen Frames in Chained Video**: Quantum alignment mismatch. Check `latent_slice_count` math and ensure `frames_to_trim` matches actual decoded overlap.
- **OOM During Generation**: Reduce resolution or switch to FP8 checkpoint (CUDA only). On MPS, the system auto-caps to full-precision.
- **Pipeline Not Switching**: The `ModelManager` caches the pipeline type. If the pipeline type string doesn't match exactly (`"ti2vid"` vs `"TI2Vid"`), a stale pipeline may be reused.
- **Audio Not Generated**: Audio generation is conditional on the pipeline supporting it and the user requesting it. Check pipeline args.

## 10. Glossary

- **Dual-Stream Transformer**: Architecture where video and audio tokens have separate Feed-Forward Networks but share the Self-Attention mechanism, enforcing temporal synchronization.
- **Distilled LoRA**: A Low-Rank Adaptation module trained to approximate the behavior of a multi-step model in fewer steps (8 vs 50).
- **Flow Matching**: Training paradigm where the model learns the vector field that transports noise to data along straight paths.
- **Quantum Alignment**: Milimo-specific term for the process of aligning pixel-space overlap to the latent grid to ensure smooth handoff between chained video chunks.
- **IC-LoRA (Image Conditioning LoRA)**: A LoRA variant that embeds a specific visual concept (person, character, object) into the model for consistent reproduction.
- **Two-Stage Pipeline**: First stage generates at low resolution, second stage upsamples spatially and/or temporally.
