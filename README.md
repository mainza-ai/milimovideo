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
- **AI Core**: **LTX-2 19B** (Lightricks), **Gemma 3** (Google DeepMind) for prompt understanding, **Diffusers** for pipeline management.
- **Optimization**: Supports **FP8 on CUDA** and heavily optimized for **Apple Silicon (MPS)**.

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
git clone https://github.com/yourusername/milimo-video.git
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
    pip install -r requirements.txt
    ```

2.  **Download Models**:
    Place the following into `LTX-2/models/checkpoints/`:
    - [LTX-2 19B Distilled](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors)
    - [LTX-2 Spatial Upscaler](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors)
    - [Gemma 3 Text Encoders](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) (place contents in `gemma-3-12b-it-qat-q4_0-unquantized/`)

### 3. Running the Studio

**Start the Backend API**:
```bash
# In a new terminal
source milimov/bin/activate
cd backend
python server.py
```

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
