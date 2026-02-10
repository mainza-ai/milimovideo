# Milimo Video — System Architecture

## 1. System Overview

Milimo Video is a local-first, AI-native cinematic studio with a split architecture:

- **Frontend**: React 18 / Vite / TypeScript — UI, Timeline State, Multi-Track NLE, Preview Player
- **Backend**: Python / FastAPI — Model Inference, Asset Management, Job Queue, Database
- **AI Models**: LTX-2 (Video), Flux 2 (Image/Inpainting), SAM 3 (Segmentation)
- **Microservice**: SAM 3 runs as a separate FastAPI server on port 8001

## 2. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend — React / Vite / TypeScript"
        App[App.tsx] --> Layout[Layout.tsx]
        Layout --> Player[CinematicPlayer]
        Layout --> VT[VisualTimeline]
        Layout --> Inspector[InspectorPanel]
        Layout --> Library[MediaLibrary]
        Layout --> Elements[ElementPanel]
        Layout --> Storyboard[StoryboardView]
        Layout --> Images[ImagesView]
        Layout --> PM[ProjectManager]
        
        Player --> PBE[PlaybackEngine]
        PBE --> GAM[GlobalAudioManager]
        PBE --> TU[timelineUtils]
        
        VT --> TTrack[TimelineTrack]
        TTrack --> TClip[TimelineClip]
        TTrack --> AClip[AudioClip]
        VT --> Playhead[Playhead]
        VT --> Snap[snapEngine]

        Inspector --> Cond[ConditioningEditor]
        Inspector --> AdvSettings[AdvancedSettings]
        Inspector --> ShotParams[ShotParameters]
        Inspector --> NarDir[NarrativeDirector]

        All["All Components"] --> Store[useTimelineStore]
        Store --> SSE_Client[SSE Event Listener]
        Store --> JobPoll[jobPoller]
    end

    subgraph "Backend — Python / FastAPI (port 8000)"
        SSE_Client <-->|"SSE / REST"| API[FastAPI Server]
        JobPoll -->|"REST GET"| API
        API --> Router[APIRouter]
        
        Router --> R_Proj[routes/projects.py]
        Router --> R_Jobs[routes/jobs.py]
        Router --> R_Assets[routes/assets.py]
        Router --> R_Elem[routes/elements.py]
        Router --> R_Story[routes/storyboard.py]
        
        R_Jobs --> Tasks
        R_Story --> Tasks
        
        API --> DB[(SQLite via SQLModel)]
        API --> EvMgr[EventManager — SSE]
    end

    subgraph "Task Engine"
        Tasks[tasks/] --> VideoTask[video.py]
        Tasks --> ChainedTask[chained.py]
        Tasks --> ImageTask[image.py]
        
        VideoTask --> ME[ModelManager]
        VideoTask -->|"num_frames==1"| FluxW[FluxInpainter]
        ChainedTask --> ME
        ChainedTask --> SBMgr[StoryboardManager]
        ImageTask --> FluxW
        
        R_Elem --> ElMgr[ElementManager]
        ElMgr --> FluxW
        R_Elem --> InpMgr[InpaintingManager]
        InpMgr --> FluxW
        InpMgr -->|"HTTP POST"| SAM_Svc
    end

    subgraph "AI Inference"
        ME --> LTX_TI2V[TI2VidTwoStagesPipeline]
        ME --> LTX_IC[ICLoraPipeline]
        ME --> LTX_KF[KeyframeInterpolationPipeline]
        
        FluxW --> Flux2_Model[Flux 2 Klein 9B]
        FluxW --> IP_Adapter[IP-Adapter — CLIP ViT-L]
        FluxW --> AE_Native[Native AutoEncoder]
        FluxW --> AE_Diffusers[Diffusers AE Fallback]
        
        LTX_TI2V --> GPU[(GPU / MPS)]
        LTX_IC --> GPU
        LTX_KF --> GPU
        Flux2_Model --> GPU
    end

    subgraph "SAM 3 Microservice (port 8001)"
        SAM_Svc[FastAPI Server] --> SAM_Model[SAM 3 Model]
        SAM_Svc --> SAM_Health["/health"]
        SAM_Svc --> SAM_Predict["/predict/mask"]
    end

    API -->|Static Files| FS[File System]
```

## 3. Core Component Interactions

| Interaction | Protocol | Description |
|---|---|---|
| **UI → Backend** | HTTP REST | CRUD for projects, shots, elements, assets. Job dispatch. |
| **Backend → UI** | SSE (Server-Sent Events) | Real-time progress updates, job completion, error logs. |
| **UI Refresh Sync** | HTTP GET `/status/{job_id}` | One-shot job state recovery on page load via `jobPoller`. |
| **Worker → LTX-2** | Direct Python Import | `ModelManager` loads LTX-2 pipelines into VRAM. One pipeline cached at a time. |
| **Worker → Flux 2** | Direct Python Import | `FluxInpainter` singleton (persistent in memory) for image gen + inpainting. |
| **Backend → SAM** | HTTP REST (port 8001) | Segmentation mask generation via SAM 3 microservice (multipart POST). |
| **Timeline → Player** | Headless rAF subscription | `PlaybackEngine` drives `currentTime` via `requestAnimationFrame`. `CinematicPlayer` subscribes. |
| **Playback → Audio** | `GlobalAudioManager` singleton | Manages `HTMLAudioElement` pool, sync/tick pattern. |
| **Flux AE Hot-Swap** | Internal | `FluxInpainter` toggles between native AE (supports reference conditioning) and diffusers AE (fallback) based on `enable_ae` flag. |

## 4. Key Subsystems

### A. The Timeline Engine (Frontend)
CSS-based NLE — no canvas rendering. Uses absolute positioning for performance.

| Component | Role |
|---|---|
| `timelineStore.ts` | Single Zustand store (7 slices) — source of truth |
| `VisualTimeline.tsx` | Renders tracks, clips, playhead. Handles drag-and-drop. |
| `TimelineTrack.tsx` | Renders a single track with clips. |
| `TimelineClip.tsx` | Individual clip with drag/resize handles. Uses `translateX` for GPU compositing. |
| `AudioClip.tsx` | Specialized clip for audio tracks with waveform-like display. |
| `PlaybackEngine.tsx` | Headless rAF loop driving `currentTime` and audio sync. |
| `timelineUtils.ts` | `computeTimelineLayout()` — centralized layout with magnetic V1 logic. |
| `snapEngine.ts` | Snap-to-grid/clip/playhead during drag operations. |

### B. The Generation Pipeline (Backend)
Handles video, image, chained, and inpainting generation.

| Component | Role |
|---|---|
| `model_engine.py` | `ModelManager` singleton — loads/manages LTX-2 pipelines. Auto-selects checkpoint (full > FP8). |
| `tasks/video.py` | `generate_video_task()` — orchestrates pipeline selection, prompt enhancement, path resolution, delegates to standard or chained gen. Single-frame requests shortcut to Flux 2. |
| `tasks/chained.py` | `generate_chained_video_task()` — autoregressive multi-chunk generation with Quantum Alignment latent handoff and ffmpeg overlap trimming. |
| `tasks/image.py` | `generate_image_task()` — Flux 2 image generation with element/trigger resolution, IP-Adapter reference images, Asset DB record creation. |
| `models/flux_wrapper.py` | `FluxInpainter` singleton — wraps Flux 2 Klein 9B with AE hot-swap, MPS hacks, IP-Adapter, True CFG double-pass, and RePaint inpainting. |
| `storyboard/manager.py` | `StoryboardManager` — dual-mode (worker/server). Chunk preparation, last-frame extraction via ffmpeg, shot-based prompt enrichment with narrative context. |

### C. The API Layer (Backend)
5 routers with ~30 endpoints.

| Router | Prefix | Key Endpoints |
|---|---|---|
| `projects.py` | `/projects` | CRUD, save, render, split shot, get images |
| `jobs.py` | `/status`, `/generate` | Status polling, advanced generation, image generation, cancel |
| `assets.py` | `/uploads`, `/assets` | File upload, list media, delete, get last frame |
| `elements.py` | `/elements`, `/edit` | Element CRUD, visualize, inpaint, segment |
| `storyboard.py` | `/storyboard` | Parse script, commit storyboard, get hierarchy, generate shot |

### D. The Player System (Frontend)
"Program Monitor" pattern with multi-component composition.

| Component | Role |
|---|---|
| `CinematicPlayer.tsx` | Container with `VideoSurface`, `PlayerHUD`, `LoadingOverlay`, `ControlsBar` |
| `VideoSurface` | `memo`'d + `forwardRef` `<video>` element with drift correction |
| `PlayerHUD` | Resolution/FPS/Seed overlay |
| `LoadingOverlay` | Generation progress display with ambient glow effects |
| `ControlsBar` | Play/pause, edit mode toggle, fullscreen |

### E. The Inspector System (Frontend)
Shot editing panel with sub-components.

| Component | Role |
|---|---|
| `InspectorPanel.tsx` | Master panel — shows selected shot details, generate/cancel/extend |
| `ShotParameters.tsx` | Resolution, frames, FPS, seed controls |
| `AdvancedSettings.tsx` | CFG scale, pipeline override, enhance prompt toggle, enable AE, True CFG |
| `ConditioningEditor.tsx` | Image/video conditioning management with drag-drop |
| `NarrativeDirector.tsx` | Action/dialogue/character metadata for storyboard |

### F. The SAM 3 Microservice
Isolated segmentation service for inpainting workflows.

| Component | Role |
|---|---|
| `sam3/start_sam_server.py` | FastAPI server (port 8001). Loads SAM 3 at startup with `build_sam3_image_model`. |
| `GET /health` | Returns `{status, model_loaded}` for readiness probes |
| `POST /predict/mask` | Accepts image (multipart) + points (JSON) → returns binary PNG mask |
| `InpaintingManager` | Backend HTTP client — sends image+points to SAM, saves mask, delegates to FluxInpainter for RePaint inpainting |
