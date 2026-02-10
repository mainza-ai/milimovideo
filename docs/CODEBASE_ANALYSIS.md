# Milimo Video — Codebase Analysis

**Date:** 2026-02-09  
**Version:** 0.3.0-Analysis

## Executive Summary

Milimo Video is a **local-first, AI-native cinematic studio** combining a React/Zustand frontend with a Python/FastAPI backend. The system orchestrates three AI models — **LTX-2** (video generation), **Flux 2** (image generation/inpainting), and **SAM 3** (segmentation) — through a multi-track Non-Linear Editor (NLE) interface.

The codebase spans **~55 source files** across two primary subsystems: a 24-file Python backend and a 30+ file TypeScript/React frontend, connected via REST endpoints and Server-Sent Events (SSE). A third subsystem — the SAM 3 microservice — runs independently on port 8001.

## Documentation Artifacts

| Document | Description |
|---|---|
| **[01_system_architecture.md](./01_system_architecture.md)** | Architecture diagrams, component overview, subsystem interactions, and the SAM 3 microservice |
| **[02_data_models.md](./02_data_models.md)** | ER diagrams for all 6 database tables, Pydantic schemas, TypeScript types, and storage layout |
| **[03_ai_pipelines.md](./03_ai_pipelines.md)** | LTX-2 (3 pipeline types + chained gen), Flux 2 (FluxInpainter with IP-Adapter, True CFG, RePaint), SAM 3 segmentation & tracking |
| **[04_frontend_state.md](./04_frontend_state.md)** | Zustand store architecture — 7 slices, 30+ components, optimization strategies |
| **[05_execution_flow.md](./05_execution_flow.md)** | Sequence diagrams for generation, SSE events, startup, storyboard, chained gen, and inpainting flows |
| **[06_file_dependency.md](./06_file_dependency.md)** | File dependency graphs for backend and frontend with complete import maps |

### AI Model Reference Documents
| Document | Description |
|---|---|
| **[flux2-bible.md](./flux2-bible.md)** | FLUX.2 repository deep-dive: model architecture, sampling, and full Milimo integration analysis |
| **[ltx2-bible.md](./ltx2-bible.md)** | LTX-2 repository deep-dive: dual-stream transformer, pipeline types, and chained gen with Quantum Alignment |
| **[sam3-bible.md](./sam3-bible.md)** | SAM 3 deep-dive: Sam3Processor, text/point/box segmentation, video tracking, microservice endpoints |

## Key Architectural Decisions

### 1. Multi-Track NLE with Magnetic V1
The timeline supports 3 tracks (V1 Main, V2 Overlay, A1 Audio). **Track 0 (V1)** uses "magnetic" sequential placement — shots stack end-to-end automatically. Tracks V2/A1 use **free placement** via `startFrame`. This is centralized in `computeTimelineLayout()`.

### 2. Singleton Model Managers
- **`ModelManager`** (in `model_engine.py`) holds a single LTX-2 pipeline at a time. It supports 3 pipeline types (`ti2vid`, `ic_lora`, `keyframe`), swapping between them with explicit VRAM cleanup (`gc.collect()` + device cache clear).
- **`FluxInpainter`** (in `models/flux_wrapper.py`) is a separate singleton wrapping Flux 2 Klein 9B. It persists the flow model, autoencoder, text encoder (Qwen 3), and optionally the IP-Adapter (CLIP ViT-L) in memory. Supports AE hot-swap between native and diffusers variants.

### 3. The "God Store" Pattern
The frontend relies on a single `useTimelineStore` composed from **7 slices** (Project, Shot, Playback, UI, Track, Element, Server). Performance is maintained through granular selectors, `useShallow`, `transientDuration`, and headless subscriptions.

### 4. SSE + Job Polling Hybrid
Real-time progress uses SSE (`EventManager` → `ServerSlice.handleServerEvent`). On page refresh, `jobPoller.ts` performs a one-shot sync to recover in-flight job state from the database.

### 5. Storyboard → Timeline Pipeline
Scripts are parsed via `ScriptParser` (regex-based screenwriting format) into `Scene` → `Shot` hierarchies, which then map to timeline clips for generation.

### 6. SAM 3 as Isolated Microservice
SAM 3 runs on port 8001 as a standalone FastAPI server (`sam3/start_sam_server.py`). It supports text-prompted segmentation (`Sam3Processor`), click-to-segment (`inst_interactive_predictor`), and video object tracking (`Sam3VideoPredictor`, lazy-loaded with MPS/CUDA/CPU device auto-detection). The main backend communicates with it via HTTP POST through `InpaintingManager` and `TrackingManager`. Inpaint jobs are persisted as `Job` DB records for status polling via `/status/{job_id}`.

### 7. MPS-First Development
The codebase is designed to run on Apple Silicon (MPS) with CUDA as the primary target. Both LTX-2 and Flux 2 wrappers include device-specific hacks:
- VAE decode forced to CPU + float32 (Flux) or float32-only (LTX)
- FP8 disabled on MPS
- Transformer forced to float32 dtype
- Memory management via `torch.mps.empty_cache()`

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend Framework** | React 18 + TypeScript |
| **State Management** | Zustand with `persist` + `zundo` (undo/redo) |
| **Bundler** | Vite |
| **Backend Framework** | FastAPI (Python 3.10+) |
| **Database** | SQLite via SQLModel (SQLAlchemy) |
| **Real-time** | Server-Sent Events (SSE via `sse-starlette`) |
| **Video AI** | LTX-2 (19B Dual-Stream Transformer, three pipelines + chained gen) |
| **Image AI** | Flux 2 Klein 9B (FluxInpainter with IP-Adapter, True CFG, RePaint inpainting) |
| **Segmentation AI** | SAM 3 (microservice on port 8001, `Sam3Processor` + `inst_interactive_predictor` + `Sam3VideoPredictor` with MPS/CUDA/CPU device support) |
| **Video Processing** | FFmpeg (thumbnails, frame extraction, encoding, overlap trimming, concat) |
| **Animation** | Framer Motion |

## File Inventory

### Backend (`backend/`) — 24 source files
```
server.py              config.py              database.py
worker.py              job_utils.py           events.py
schemas.py             model_engine.py        file_utils.py
cleanup_assets.py      migrate_v1_v2.py       
routes/__init__.py     routes/projects.py     routes/jobs.py
routes/assets.py       routes/elements.py     routes/storyboard.py
tasks/video.py         tasks/chained.py       tasks/image.py
managers/element_manager.py
managers/inpainting_manager.py
managers/tracking_manager.py
services/script_parser.py
storyboard/manager.py
models/flux_wrapper.py
```

### SAM 3 Microservice (`sam3/`) — 1 entry point
```
start_sam_server.py    # FastAPI on port 8001 (8 endpoints: health, predict, detect, segment, track)
sam3/                  # SAM 3 package (build_sam3_image_model, Sam3Processor, Sam3VideoPredictor)
```

### Frontend (`web-app/src/`) — 30+ source files
```
App.tsx                config.ts              main.tsx
components/Layout.tsx  components/Controls.tsx
components/Player/CinematicPlayer.tsx
components/Player/PlaybackEngine.tsx
components/Editor/TrackingPanel.tsx
components/Editor/MaskingCanvas.tsx
components/Timeline/VisualTimeline.tsx
components/Timeline/TimelineTrack.tsx
components/Timeline/TimelineClip.tsx
components/Timeline/AudioClip.tsx
components/Timeline/Playhead.tsx
components/Timeline/TimeDisplay.tsx
components/Inspector/InspectorPanel.tsx
components/Inspector/AdvancedSettings.tsx
components/Inspector/ConditioningEditor.tsx
components/Inspector/NarrativeDirector.tsx
components/Inspector/ShotParameters.tsx
components/Library/MediaLibrary.tsx
components/Library/ElementManager.tsx
components/Library/ElementPanel.tsx
components/Images/ImagesView.tsx
components/Storyboard/StoryboardView.tsx
components/ProjectManager.tsx
components/MediaUploader.tsx
components/VideoPlayer.tsx
components/ErrorBoundary.tsx
components/Toggle.tsx
stores/timelineStore.ts  stores/types.ts
stores/slices/projectSlice.ts  stores/slices/shotSlice.ts
stores/slices/playbackSlice.ts stores/slices/uiSlice.ts
stores/slices/trackSlice.ts    stores/slices/elementSlice.ts
stores/slices/serverSlice.ts
utils/GlobalAudioManager.ts  utils/jobPoller.ts
utils/snapEngine.ts          utils/timelineUtils.ts
```
