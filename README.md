# Milimo Video

<div align="center">

![Milimo Video](web-app/public/logo.png)

**The AI-Native Cinematic Studio**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Frontend](https://img.shields.io/badge/Frontend-React_Vite-61DAFB.svg)](web-app/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI_Python-009688.svg)](backend/)
[![Model](https://img.shields.io/badge/Model-LTX--2_19B-purple.svg)](https://github.com/Lightricks/LTX-2)

</div>

---

**Milimo Video** is a state-of-the-art, open-source AI video production studio designed for filmmakers, not just prompt engineers. It wraps the powerful [LTX-2 (19B)](https://github.com/Lightricks/LTX-2) foundation model in a professional non-linear editing (NLE) interface, allowing you to craft consistent, long-form video narratives with unprecedented control.

## ✨ Key Features

### 🎬 Narrative Director & Intelligent Co-Pilot
Milimo isn't just a generation slot machine; it understands your story.
- **Narrative Director**: An autonomous agent that enhances your prompts using Gemma 3, adding visual richness while adhering to your creative intent. It evolves continuously as you extend shots.
- **Smart Continue**: Autoregressive video chaining that maintains character and style consistency across long generations.
- **Live Evolution**: Watch the "Director's" thought process in real-time as it crafts the scene description during generation.

### 📝 Storyboard Engine (New)
Transform screenplays into video productions instantly.
- **Script-to-Video**: Paste your standard screenplay text, and Milimo parses it into distinct Scenes and Shots.
- **Story Elements**: Create reusable **Characters, Locations, and Objects** in the new **Elements Panel**.
- **Auto-Injection**: The system automatically detects `@Element` references in your script and injects their detailed visual descriptions into the prompt, ensuring consistency across shots.
- **Dual View Modes**: Switch seamlessly between the **Timeline** (NLE) for editing and the **Storyboard** for high-level narrative planning.

### 🎞️ Professional Non-Linear Editor (NLE)
A fully functional timeline built for the AI workflow.
- **Multi-Track Editing**: Compose complex scenes with multiple video and audio tracks (roadmap).
- **Undo / Redo System**: Experiment fearlessly with full state history management (powered by `zundo`).
- **Precision Control**: Frame-accurate seeking, scrubbing, and trimming.
- **Extend & Morph**: Seamlessly extend clips or morph between shots using advanced conditioning.

### 🌟 High Fidelity Generation (New)
Pushing the boundaries of LTX-2.
- **4K Ultra HD**: Generate stunning visuals at **3840x2160** resolution.
- **Dynamic High FPS**: Native support for **50fps and 60fps** playback for buttery smooth motion.
- **Deep Context**: Generate up to **20 seconds** in a single continuous shot.
- **Audio Intelligence**: The AI Co-Pilot now directs audio soundscapes, generating synchronized ambient noise and dialogue descriptions.

### ✂️ In-Painting & Editing (Completed)
Professional-grade retouching and alterations.
- **Flux 2 In-Painting**: Select any frame, mask an area, and use natural language to magically add or remove elements (e.g., "Add sunglasses", "Remove car").
- **Integrated Mask Editor**: Draw precise masks directly on your video frames within the player interface.
- **SAM 3 Integration**: (Optional) Use Segment Anything Model 3 for intelligent, click-based object masking.

### 🧠 Advanced Conditioning & Control
Go beyond simple text-to-video.
- **Image-to-Video**: Bring static images to life with motion.
- **Video-to-Video**: Restyle existing footage or use it as a structure reference (IC-LoRA).
- **Keyframe Interpolation**: Define the start and end frames, and let Milimo hallucinate the journey between them.
- **Custom LoRAs**: Support for loading external LoRA adapters for specific styles or characters.

### ⚡ Local & Privately Hosted
- **100% Offline Capable**: Runs entirely on your local hardware (NVIDIA GPU or Apple Silicon).
- **SQLite Persistence**: Robust, single-file database for all your projects and assets.
- **No API Costs**: You own the model and the generations.
- **Privacy First**: Your creative assets never leave your machine.

---

## 🛠️ Architecture

Milimo Video is built on a modern, robust stack:

- **Frontend**: **React 18**, **TypeScript**, **Vite**, **Zustand** (State), **TailwindCSS** (Styling). Designed with a premium "Glassmorphism" aesthetic.
- **Backend**: **FastAPI** (Python 3.10+), **SQLAlchemy** (SQLite), **AsyncIO** worker queues.
- **Microservices**: **SAM 3** runs as a dedicated decoupled service (Port 8001) to ensure environment stability.
- **AI Core**: 
    - **LTX-2 19B** (Lightricks) for video generation.
    - **Gemma 3** (Google DeepMind) for prompt understanding.
    - **Flux 2** (Black Forest Labs) for high-fidelity in-painting.
- **Optimization**: 
    - Supports **FP8 on CUDA**.
    - Heavily optimized for **Apple Silicon (MPS)**, including specific VAE stability fixes (Float32 forcing) to prevent black images.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** (Required for video encoding/decoding)
- **High-End GPU**:
  - **NVIDIA**: 16GB+ VRAM recommended (runs on 12GB with heavy offloading).
  - **Apple Silicon**: M1/M2/M3 Max or Ultra recommended (32GB+ RAM).

### 1. Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/mainza-ai/milimovideo.git
cd milimovideo
```

### 2. Backend Setup & Model Download

Milimo uses a specialized version of LTX-2. You must download the checkpoints manually.

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

    **SAM 3 Setup (In-Painting):**
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

**1. Start the Backend API (Core)**:
```bash
# In a new terminal
source milimov/bin/activate
cd backend
python server.py
```

**2. Start the SAM 3 Service (Optional)**:
```bash
# In a separate terminal
conda activate sam3_env
cd sam3
python start_sam_server.py
```
*Note: This must be running for In-Painting features to work.*

**Start the Web Interface**:
```bash
# In a separate terminal
cd web-app
npm install
npm run dev
```

Visit **`http://localhost:5173`** to enter the studio.

---

## 🎮 Controls

- **Space**: Play / Pause
- **Cmd + Z**: Undo
- **Cmd + Shift + Z**: Redo
- **S / Cmd + S**: Save Project
- **Delete / Backspace**: Remove selected shot
- **Drag & Drop**: Import images/videos from file explorer directly onto the timeline or conditioning slots.

---

## 📜 License

This project is licensed under the Apache 2.0 License.
Based on [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks.
