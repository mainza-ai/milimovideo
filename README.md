# Milimo Video

<div align="center">

![Milimo Video](web-app/public/logo.png)

**The AI-Native Cinematic Studio**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Frontend](https://img.shields.io/badge/Frontend-React_18_Vite-61DAFB.svg)](web-app/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI_Python-009688.svg)](backend/)
[![Models](https://img.shields.io/badge/Models-LTX--2_|_Flux.2_|_SAM_3-purple.svg)](https://github.com/Lightricks/LTX-2)

</div>

---

**Milimo Video** is a state-of-the-art, open-source AI video production studio designed for filmmakers. It unifies the world's best foundation models into a cohesive, professional workflow — running entirely **local-first** on your own machine.

Unlike simple "prompt-to-video" interfaces, Milimo is a full **Non-Linear Editor (NLE)** that combines:
*   **LTX-2 19B** — Dual-stream transformer for cinematic video generation (text-to-video, image-to-video, keyframe interpolation).
*   **Flux 2 Klein 9B** — High-fidelity image synthesis with IP-Adapter reference conditioning and RePaint inpainting.
*   **SAM 3** — Text-prompted segmentation, click-to-segment, and video object tracking via standalone microservice.
*   **Gemma 3** — Intelligent prompt enhancement and narrative direction.

## ✨ Key Features

### 🎬 Visual Conditioning & Character Consistency
Achieve what standard models can't: persistent identities across shots.
- **IP-Adapter Integration**: Flux 2's IP-Adapter (CLIP ViT-L → 4-token projection) injects visual style and character identity directly into the generation latent space.
- **Reference Conditioning**: Native AE encodes reference images with temporal offsets, concatenating them to the denoising input for style-faithful generation.
- **Story Elements**: Define reusable **Characters, Locations, and Objects** with trigger words (e.g., `@Hero`). The system auto-detects triggers in prompts and injects the correct IP-Adapter images and enriched text.
- **Projected Latents**: Support for projecting reference images into LTX-2's latent space for seamless Image-to-Video transitions.

![Milimo Elements](assets/milimo_video_elements.png)

### 📝 Storyboard Engine
Transform screenplays into video productions instantly.
- **Script-to-Video**: Paste standard screenplay text — Milimo parses it into Scenes and Shots via regex-based `ScriptParser`.
- **Auto-Injection**: `ElementManager` scans for `@Element` references and injects visual/textual conditioning + narrative context (action, dialogue, character).
- **Chained Generation**: Shots exceeding 121 frames auto-trigger **Quantum Alignment** — autoregressive chunk-by-chunk generation with latent handoffs aligned to the 8-pixel VAE grid, ensuring seamless visual continuity.

### 🎞️ Professional Non-Linear Editor (NLE)
A fully functional timeline built for the AI workflow.
- **Multi-Track Editing**: 3 tracks — V1 (magnetic main), V2 (overlay, free placement), A1 (audio).
- **Smart Continue**: Autoregressive video chaining via `StoryboardManager` with last-frame extraction and overlap trimming.
- **Precision Control**: Frame-accurate seeking, scrubbing, trimming (`trimIn`/`trimOut`), and `snapEngine` snapping.
- **CSS-Based Timeline**: GPU-accelerated clip positioning (`translateX`), granular Zustand selectors, and `useShallow` for 60fps UI responsiveness.

![Milimo Timeline](assets/milimo_video_timeline.png)

### ✂️ In-Painting & Intelligent Editing
Professional-grade retouching powered by the SAM 3 → Flux 2 pipeline.
- **Flux RePaint Inpainting**: Select any frame, mask an area, and use natural language to add or remove elements. Uses iterative mask-blended denoising with real-time SSE progress. Inpaint jobs are persisted to the database for reliable status polling via `/status/{job_id}`.
- **SAM 3 Text-Prompted Segmentation**: Describe what to segment ("a person", "the sky") — SAM 3 finds all matching objects with bounding boxes and confidence scores.
- **Click-to-Segment**: Click on objects in the video frame for instant SAM 3-powered mask generation. No manual roto-scoping.
- **Video Object Tracking**: Select an object on one frame → SAM 3 tracks it across every frame of the video, bidirectionally. Full UI via `TrackingPanel` with session lifecycle management (start → prompt → propagate → navigate results).
- **AE Hot-Swap**: Toggle between native AutoEncoder (supports reference conditioning) and diffusers AE fallback via the `enable_ae` flag.
- **True CFG Mode**: Optional double-pass negative prompting for Flux 2 (2× inference time, disabled by default).

![Milimo Image Generation](assets/milimo_video_image_generation.png)

### 🧠 Advanced Generation
- **Dual-Stage Pipeline**: LTX-2 generates at half-res, then spatially upsamples 2× with distilled LoRA-384.
- **3 Pipeline Modes**: `ti2vid` (text/image-to-video), `ic_lora` (IC-LoRA conditioning), `keyframe` (keyframe interpolation).
- **Single-Frame Shortcut**: When `num_frames==1`, video gen silently delegates to Flux 2 for instant image generation.
- **Real-Time Progress**: SSE (Server-Sent Events) stream denoising step progress, ETA, and enhanced prompts to the UI in real-time.

---

## 🛠️ Architecture

![Milimo Architecture](assets/Milimo_Video_Studio_Architecture_infograph.png)


Milimo Video is built on a modern, robust stack:

| Layer | Technology |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Zustand (7-slice store + persist + zundo undo/redo) |
| **Backend** | FastAPI (Python 3.10+), SQLModel/SQLAlchemy (SQLite), SSE via `sse-starlette` |
| **Video AI** | LTX-2 19B Dual-Stream Transformer — 3 pipelines + chained generation |
| **Image AI** | Flux 2 Klein 9B (`FluxInpainter`) — IP-Adapter, True CFG, RePaint inpainting |
| **Segmentation** | SAM 3 Microservice (port 8001) — `Sam3Processor` (text/box), `inst_interactive_predictor` (click), `Sam3VideoPredictor` (tracking, MPS/CUDA/CPU) |
| **Prompt AI** | Gemma 3 (via LTX-2 text encoder) — cinematic prompt enhancement |
| **Processing** | FFmpeg — thumbnails, frame extraction, overlap trimming, concat |

### MPS-First Optimization
Designed for Apple Silicon with CUDA as primary target:
- FP8 on CUDA, float32 fallback on MPS
- VAE decode CPU-offloaded on MPS (prevents black output)
- Transformer forced to float32 dtype on MPS
- Memory managed via `gc.collect()` + `torch.mps.empty_cache()`
- SAM 3 Video Predictor: device auto-detection (CUDA → MPS → CPU), guarded `torch.cuda.*` calls
- `PYTORCH_ENABLE_MPS_FALLBACK=1` for SAM 3 ops not yet on MPS

### Documentation
See the [`docs/`](docs/) directory for comprehensive technical documentation:
- [System Architecture](docs/01_system_architecture.md) — Component diagrams and subsystem interactions
- [Data Models](docs/02_data_models.md) — ER diagrams, Pydantic schemas, TypeScript types
- [AI Pipelines](docs/03_ai_pipelines.md) — LTX-2, Flux 2, SAM 3 pipeline analysis with flow diagrams
- [Frontend State](docs/04_frontend_state.md) — Zustand store architecture and optimization strategies
- [Execution Flow](docs/05_execution_flow.md) — Sequence diagrams for all generation workflows
- [File Dependencies](docs/06_file_dependency.md) — Import graphs and module analysis
- [Flux 2 Bible](docs/flux2-bible.md) — Deep-dive into Flux 2 integration
- [LTX-2 Bible](docs/ltx2-bible.md) — Deep-dive into LTX-2 integration
- [SAM 3 Bible](docs/sam3-bible.md) — Deep-dive into SAM 3 microservice

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg**
- **High-End GPU**:
  - **NVIDIA**: 16GB+ VRAM recommended.
  - **Apple Silicon**: M1/M2/M3/M4 Max or Ultra recommended (32GB+ RAM).

### 1. Installation

Clone the repository:
```bash
git clone https://github.com/mainza-ai/milimovideo.git
cd milimovideo
```

### 2. Backend Setup

Milimo uses a specialized environment for LTX-2 and Flux.

1.  **Create Environment**:
    ```bash
    python3 -m venv milimov
    ./milimov/bin/pip install -e ./LTX-2/packages/ltx-core
    ./milimov/bin/pip install -e ./LTX-2/packages/ltx-pipelines
    ./milimov/bin/pip install -e ./flux2
    ./milimov/bin/pip install -r backend/requirements.txt
    ```

2.  **Download LTX-2 Models**:
    Place the following into `LTX-2/models/checkpoints/`:
    - [LTX-2 19B Distilled](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors)
    - [LTX-2 Distilled LoRA-384](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors) → `checkpoints/`
    - [LTX-2 Spatial Upscaler](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) → `upscalers/`
    - [Gemma 3 Text Encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) → `text_encoders/gemma3/`

3.  **Download Flux 2 Models**:
    Place files in `backend/models/flux2/`:

    ```text
    backend/models/flux2/
    ├── flux-2-klein-9b.safetensors    # Flow model (9B params)
    ├── ae.safetensors                 # Native AutoEncoder (preferred)
    ├── vae/                           # Diffusers AE fallback (config.json + diffusion_pytorch_model.safetensors)
    ├── text_encoder/                  # Qwen 3 (8B) text encoder
    ├── tokenizer/                     # Qwen tokenizer files
    └── ip-adapter.safetensors         # IP-Adapter weights (CLIP ViT-L projection)
    ```

4.  **SAM 3 Setup (Segmentation & Tracking)**:
    The SAM 3 service runs in a separate environment (`sam3_env`).
    1.  Create environment:
        ```bash
        conda create -n sam3_env python=3.12
        conda activate sam3_env
        pip install -e sam3
        pip install fastapi uvicorn python-multipart psutil pycocotools huggingface_hub
        ```
    2.  **Download Model**:
        Download the SAM 3 checkpoint from [HuggingFace](https://huggingface.co/facebook/sam3) (auto-downloads on first start if missing).
    3.  Place it in: `backend/models/sam3/sam3.pt`

### 3. Running the Studio

**1. Start the Backend API** (port 8000):
```bash
./run_backend.sh
```

**2. Start the SAM 3 Service** (port 8001, optional — for segmentation, masking & tracking):
```bash
./run_sam.sh
```

**3. Start the Web Interface** (port 5173):
```bash
./run_frontend.sh
```

Visit **`http://localhost:5173`** to enter the studio.

---

## 🎮 Controls

| Key | Action |
|---|---|
| **Space** | Play / Pause |
| **Cmd + Z** | Undo |
| **Cmd + Shift + Z** | Redo |
| **S / Cmd + S** | Save Project |
| **Delete / Backspace** | Remove selected shot |
| **Drag & Drop** | Import images/videos onto timeline |
| **✏️ Edit icon** | Toggle masking/inpainting mode |
| **⌖ Crosshair icon** | Toggle video object tracking mode |

---

---

## 🤝 Community & Collaboration

Milimo Video is an open-source project, and we welcome contributions! Whether you're a developer, filmmaker, or AI enthusiast, there are many ways to get involved:

*   **Code**: Submit Pull Requests for new features, bug fixes, or performance improvements.
*   **Models**: Share custom LoRAs or fine-tunes that work well with the LTX-2/Flux pipeline.
*   **Feedback**: Open Issues for bugs or feature requests on our [GitHub repository](https://github.com/mainza-ai/milimovideo).

Let's build the future of AI cinema together.

## ✍️ Author

**Mainza Kangombe**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/mainza-kangombe-6214295)

---

## 📜 License

This project is licensed under the Apache 2.0 License.