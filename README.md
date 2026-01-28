# Milimo Video

Milimo Video is an AI-powered cinematic video creation studio that leverages the [LTX-2](https://github.com/Lightricks/LTX-2) foundation model. It provides a non-linear editor interface for composing shots, generating video from text/images, and sequencing them into full stories.

## Features

- **Cinematic Timeline**: Non-linear video editor with multi-track support.
- **AI Generation**: Generate video shots using LTX-2 (Text-to-Video, Image-to-Video).
- **Interactive Playhead**: Scrub, play, and preview shots in real-time.
- **Drag & Drop**: Easily manage assets and drop them onto the timeline.
- **Project Management**: Save and load multiple projects.
- **Inspector**: Fine-tune generation parameters (Seed, Steps, Guidance).

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** (Required for video processing)
- **NVIDIA GPU** with 24GB+ VRAM (Recommended for LTX-2 19B model)

## Installation

### 1. Setup LTX-2
This project includes LTX-2 as a submodule (or copy). You must download the model checkpoints manually due to their size.

Follow the instructions in `LTX-2/README.md` to download the following into `LTX-2/models/checkpoints/`:
- `ltx-2-19b-distilled.safetensors`
- `ltx-2-19b-distilled-lora-384.safetensors`
- Text Encoders (Gemma 3)

### 2. Backend Setup
The backend is a FastAPI server that manages the LTX-2 inference and project state.

```bash
cd backend
python3 -m venv ../milimov
source ../milimov/bin/activate

# Install LTX-2 dependencies
pip install -e ../LTX-2

# Install Backend dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
The frontend is a React + Vite application.

```bash
cd web-app
npm install
```

## Running the Application

### 1. Start Backend
In a terminal:
```bash
source milimov/bin/activate
cd backend
# Run the server
python server.py
```
Server runs on: `http://localhost:8000`

### 2. Start Frontend
In a new terminal:
```bash
cd web-app
npm run dev
```
Open your browser to the URL shown (usually `http://localhost:5173`).

## Project Structure

- `backend/`: FastAPI server & worker logic.
- `web-app/`: React/TypeScript frontend.
- `LTX-2/`: LTX-2 Model code and pipelines.
- `milimov/`: Python Virtual Environment (not in repo).

## License
[Add License Here]
