# Milimo Video

<div align="center">

![Milimo Video](web-app/public/logo.png)

**The AI-Native Cinematic Studio**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Frontend](https://img.shields.io/badge/Frontend-React_Vite-61DAFB.svg)](web-app/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI_Python-009688.svg)](backend/)
[![Models](https://img.shields.io/badge/Models-LTX--2_|_Flux.2_|_SAM_3-purple.svg)](https://github.com/Lightricks/LTX-2)

</div>

---

**Milimo Video** is a state-of-the-art, open-source AI video production studio designed for filmmakers. It unifies the world's best foundation models into a cohesive, professional workflow.

Unlike simple "prompt-to-video" interfaces, Milimo is a full **Non-Linear Editor (NLE)** that combines:
*   **LTX-2** for cinematic video generation.
*   **Flux 2** for high-fidelity image synthesis and visual conditioning.
*   **SAM 3** for precise object masking and tracking.
*   **Gemma 3** for intelligent narrative direction.

## ✨ Key Features

### 🎬 Visual Conditioning & Character Consistency (New)
Achieve what standard models can't: persistent identities across shots.
- **IP-Adapter Integration**: Use **Flux** to inject visual style and character identity directly into the generation process.
- **Visuals-Only Prompting**: Attach a "Character Sheet" to an Element. The system uses the *visuals* to define the look, while your text defines the action. No more "blue hair, sci-fi armor" repetition in every prompt.
- **Projected Latents**: Native support for projecting reference images into LTX-2's latent space for seamless Image-to-Video transitions.

### 📝 Storyboard Engine
Transform screenplays into video productions instantly.
- **Script-to-Video**: Paste your standard screenplay text, and Milimo parses it into distinct Scenes and Shots.
- **Story Elements**: Define reusable **Characters, Locations, and Objects**. 
- **Auto-Injection**: The system automatically detects `@Element` references (e.g., `@Hero`) and injects the correct visual/textual conditioning into the prompt.

### 🎞️ Professional Non-Linear Editor (NLE)
A fully functional timeline built for the AI workflow.
- **Multi-Track Editing**: Compose complex scenes with multiple video tracks.
- **Smart Continue**: Autoregressive video chaining that maintains context from previous clips.
- **Precision Control**: Frame-accurate seeking, scrubbing, and trimming.
- **Extend & Morph**: Seamlessly extend clips or morph between shots using advanced conditioning.

### ✂️ In-Painting & Intelligent Editing
Professional-grade retouching and alterations.
- **Flux In-Painting**: Select any frame, mask an area, and use natural language to magically add or remove elements (e.g., "Add sunglasses").
- **SAM 3 Integration**: Use the **Segment Anything Model 3** to automatically detect and mask objects with a single click. No manual roto-scoping required.
- **Windowed In-Painting**: Fix specific regions of a video while keeping the rest consistent.

### 🧠 Advanced Generation
- **4K Ultra HD**: Generate stunning visuals at **3840x2160**.
- **High FPS**: Native support for **50fps and 60fps** playback.
- **Audio Intelligence**: The AI Co-Pilot directs audio soundscapes, generating synchronized ambient noise suggestions.

---

## 🛠️ Architecture

Milimo Video is built on a modern, robust stack:

- **Frontend**: **React 18**, **TypeScript**, **Vite**, **Zustand** (State), **TailwindCSS** (Styling). Designed with a premium "Glassmorphism" aesthetic.
- **Backend**: **FastAPI** (Python 3.10+), **SQLAlchemy** (SQLite).
- **AI Core**: 
    - **LTX-2 19B** (Lightricks): Video Foundation.
    - **Flux 2** (Black Forest Labs): Image Generation & In-Painting.
    - **SAM 3** (Meta): Segmentation & Tracking.
    - **Gemma 3** (Google DeepMind): Prompt Understanding & Narrative Direction.
- **Optimization**: 
    - Supports **FP8 on CUDA**.
    - Heavily optimized for **Apple Silicon (MPS)**, including specific VAE stability fixes (Float32 forcing) to prevent black images.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg**
- **High-End GPU**:
  - **NVIDIA**: 16GB+ VRAM recommended.
  - **Apple Silicon**: M1/M2/M3 Max or Ultra recommended (32GB+ RAM).

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
    cd backend
    python3 -m venv ../milimov
    source ../milimov/bin/activate
    pip install -e ../LTX-2
    pip install -e ../flux2
    pip install -r requirements.txt
    ```

2.  **Download Models**:
    Place the following into `LTX-2/models/checkpoints/`:
    - [LTX-2 19B Distilled](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors)
    - [LTX-2 Spatial Upscaler](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors)
    - [Gemma 3 Text Encoders](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) (place contents in `gemma-3-12b-it-qat-q4_0-unquantized/`)

3.  **Flux 2 Setup (In-Painting)**:
    Download the model files from [Black Forest Labs FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main) and place them in `backend/models/flux2/`.
    
    **Required Structure:**
    ```text
    backend/models/flux2/
    ├── flux-2-klein-9b.safetensors  <-- Rename main model file to this
    ├── ae.safetensors               <-- Rename VAE file to this
    ├── text_encoder/                <-- T5/Clip encoder folder
    └── tokenizer/                   <-- Tokenizer files folder
    ```

4.  **SAM 3 Setup (In-Painting Helper)**:
    The SAM 3 service runs in a separate environment (`sam3_env`).
    1.  Create environment:
        ```bash
        conda create -n sam3_env python=3.12
        conda activate sam3_env
        pip install -e sam3
        pip install fastapi uvicorn python-multipart psutil pycocotools
        # Optional: pip install eva-decord (for video support on compatible systems)
        ```
    2.  **Download Model**:
        Download `sam3.pt` (or `sam3_large.pth`) from [ModelScope](https://www.modelscope.cn/models/facebook/sam3/files) or [HuggingFace](https://huggingface.co/facebook/sam3).
    3.  Place it in: `backend/models/sam3_large.pth`

### 3. Running the Studio

**1. Start the Backend API**:
```bash
source milimov/bin/activate
cd backend
python server.py
```

**2. Start the Web Interface**:
```bash
cd web-app
npm install
npm run dev
```

Visit **`http://localhost:5173`** to enter the studio.

---

## 🎮 Controls

- **Space**: Play / Pause
- **Cmd + Z**: Undo / Redo
- **S / Cmd + S**: Save Project
- **Delete / Backspace**: Remove selected shot
- **Drag & Drop**: Import images/videos directly onto the timeline.

---

## ✍️ Author

**Mainza Kangombe**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/mainza-kangombe-6214295)

---

## 📜 License

This project is licensed under the Apache 2.0 License.
Based on [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks.
