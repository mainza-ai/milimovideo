# Milimo Video - System Documentation

## 1. Executive Summary
**Milimo Video** is an AI-Native Cinematic Studio that integrates state-of-the-art generative models into a professional Non-Linear Editing (NLE) workflow. Unlike simple "text-to-video" interfaces, Milimo provides a structured, project-based environment where filmmakers can script, storyboard, generate, and refine video content with granular control.

**Core Capabilities:**
*   **Generative NLE**: A full timeline allowing multi-track editing, dragging, dropping, and sequencing of AI-generated clips.
*   **Pipeline Unification**: Seamlessly orchestrates **LTX-2** (Video), **Flux 2** (Image/In-painting), **SAM 3** (Segmentation), and **Gemma 3** (Prompt Enrichment).
*   **Visual Consistency**: Implements "Element" tracking and Latent Handoffs to maintain character and style consistency across generated shots.
*   **Smart Automation**: Features "Smart Continue" for autoregressive video extension and a "Storyboard Engine" that parses screenplays into shot lists.

**Intended Users:** Professional creators, filmmakers, and AI researchers requiring precise control over generative video workflows.

---

## 2. System Architecture Overview

### Architectural Style
The system follows a **Client-Server** architecture with an emphasis on **Event-Driven** communication for long-running AI tasks.

*   **Frontend**: A Single Page Application (SPA) built with React, TypeScript, and Zustand. It handles project state, timeline manipulation, and user interaction.
*   **Backend**: A monolithic FastAPI service (Python) that manages the database, orchestrates AI pipelines, and serves assets.
*   **Worker Layer**: Integrated within the backend process (for single-user mode) but designed to handle heavy GPU blocking tasks. It manages the lifecycle of large foundation models.
*   **Data Store**: SQLite ([milimovideo.db](file:///Users/mck/Desktop/milimovideo/backend/milimovideo.db)) via SQLAlchemy/SQLModel for structured metadata (Projects, Shots, Assets). Filesystem is used for heavy media storage.

### Runtime Hierarchy
1.  **Web Client (Port 5173)**: User Interface.
2.  **API Server (Port 8000)**: REST API + SSE (Server-Sent Events) for real-time progress.
3.  **SAM 3 Service**: A dedicated microservice (via Conda env) for segmentation tasks to avoid dependency conflicts.
4.  **Model Scope**: LTX-2 and Flux coexist in the main process memory, managed by a singleton [ModelManager](file:///Users/mck/Desktop/milimovideo/backend/worker.py#83-195) to handle VRAM swapping.

---

## 3. Repository Structure

```text
/
├── backend/                  # Core API and AI Logic
│   ├── managers/             # Business Logic & Model Orchestration
│   │   ├── element_manager.py    # Visual conditioning & concept tracking
│   │   ├── inpainting_manager.py # Flux/SAM in-painting coordination
│   ├── services/             # Utility Services
│   │   ├── script_parser.py      # Regex-based screenplay parser
│   ├── storyboard/           # Storyboard Engine logic
│   │   ├── manager.py            # Shot chaining & latent handoff logic
│   ├── database.py           # SQLModel schema definitions
│   ├── server.py             # FastAPI entry point & endpoints
│   ├── worker.py             # Heavy AI generation tasks (LTX-2/Flux)
│   ├── models/               # Model Wrappers & Weights
│   │   ├── flux_wrapper.py       # Interfacing with flux2 dependency
│   ├── events.py             # SSE broadcasting system
│   ├── config.py             # Configuration & sys.path injection
│   └── milimovideo.db        # SQLite Database file
├── web-app/                  # Frontend Client
│   ├── src/
│   │   ├── components/       # React UI Components
│   │   │   ├── Inspector/    # Shot editing panels
│   │   │   ├── Timeline/     # Canvas & Track logic (VisualTimeline.tsx)
│   │   ├── stores/           # Global State Management
│   │   │   ├── timelineStore.ts  # Massive Zustand store (Project/Shot Data)
│   │   └── config.ts         # Frontend config
├── LTX-2/                    # [Active Fork] Lightricks LTX-2 Inference Code (Mutable)
├── flux2/                    # [Active Fork] Flux 2 Inference Code (Mutable)
├── sam3/                     # [Active Fork] SAM 3 Inference Code (Mutable)
├── original-repos/           # [Immutable Reference] pristine upstream backups
│   ├── ltx2-orginal-repo-do-not-modify/
│   ├── flux2-orginal-repo-do-not-modify/
│   └── sam3-orginal-repo-do-not-modify/
├── mv_plan/                  # Project Planning & Tests
│   ├── final_advanced_features_plan.md
│   ├── verify_advanced_features.py
│   └── ...
└── run_sam.sh                # Helper script for SAM service
```

### 3.1 Dependency Management Strategy
The project currently employs a **Fork & Reference** pattern:
*   **Active Forks** (`LTX-2`, `flux2`, `sam3`): These directories are **part of the Milimo codebase** and can be modified to support custom features (e.g., custom VAE decoding for MPS, modified attention layers).
*   **Immutable References** (`original-repos/`): Contains the pristine, untouched code from the original upstream repositories. These serve as a backup and reference point for diffing changes.
*   **Integration**: The **Active Forks** are injected into the python path via [backend/config.py](file:///Users/mck/Desktop/milimovideo/backend/config.py).

---

### 3.2 Dual-Environment Architecture
The system is split into two distinct processes to handle conflicting dependency requirements (e.g., different CUDA/Torch versions or conflicting package versions):
1.  **Main Application** (`milimov` env): Runs the FastAPI backend (Port 8000) and React Frontend. Handles LTX-2 and Flux generation.
2.  **SAM 3 Microservice** (`sam3_env` env): Runs [sam3/start_sam_server.py](file:///Users/mck/Desktop/milimovideo/sam3/start_sam_server.py) (Port 8001). dedicated solely to segmentation tasks.
    *   *Wrapper Script*: [run_sam.sh](file:///Users/mck/Desktop/milimovideo/run_sam.sh) handles the activation of the `sam3_env` and startup.

### 3.3 Embedded Weights Strategy
Unlike standard pipelines that download weights to `~/.cache`, Milimo embeds critical model weights directly in `backend/models/`.
*   `backend/models/sam3.pt`: Explicitly loaded by the SAM service.
*   `backend/models/flux2/`: Contains the full diffusers-style directory layout (VAE, Transformer, Tokenizer).

---

### 3.4 Environment Structure
*   `milimov/`: The main Python **Virtual Environment** (venv) containing backend dependencies.
*   `packages/`: (If present) Local directory for custom packages.

---

## 4. Critical Implementation Findings & Known Issues

### 4.1 MPS Dtype Compatibility (Apple Silicon)
> [!IMPORTANT]
> **MPS works correctly with pure `bfloat16`**. Previous attempts to force `float16` or `float32` caused OOM.
>
> **Resolution (2026-02-02)**: Reverted ALL LTX-2 files to pure `torch.bfloat16`.
> Mixed precision doubled memory and caused OOM at tile 5/15.
>
> **Fix Command**:
> ```bash
> git checkout HEAD -- LTX-2/packages/ltx-core/ LTX-2/packages/ltx-pipelines/
> ```


### 4.2 MPS Memory Optimization (VAE Decode OOM Prevention)
> [!CAUTION]
> **Do not add MPS-specific dtype overrides** - Pure BFloat16 resolves memory issues.

**Problem Solved**: OOM at tile 5/15 was caused by mixed precision (float16 → float32).

**Solution**: Use pure `torch.bfloat16` with standard `TilingConfig.default()`.

| Approach | Memory | Status |
|----------|--------|--------|
| Mixed precision | ~16GB | ❌ OOM |
| Pure BFloat16 | ~8GB | ✅ Works |


### 4.3 Project-Scoped Path Structure (2026-02-02)
> [!IMPORTANT]
> **All output paths MUST use project-scoped structure**: `/projects/{project_id}/generated/...`
>
> **Legacy paths removed**: The old `/generated/...` (without project prefix) is no longer supported.

**Path Structure:**
```
/projects/{project_id}/
├── generated/           # Videos and images
│   ├── {job_id}.mp4
│   ├── {job_id}.jpg
│   └── images/{job_id}.jpg   # Flux-generated images
├── thumbnails/          # Thumbnail images
│   └── {job_id}_thumb.jpg
├── assets/              # Uploaded assets
└── workspace/           # Temp working files
```

**Key Changes:**
- `get_project_output_paths()` → **requires** `project_id`, raises `ValueError` if missing
- `generate_image_task()` → **requires** `project_id`
- All output URLs use format: `/projects/{project_id}/generated/{filename}`
- Endpoints skip/reject jobs with legacy paths


### 4.4 Embedded Weights Strategy

---

## 5. Core Concepts & Design Philosophy
Unlike stateless generation tools, Milimo uses **Projects**. A Project contains global settings (Resolution, FPS, Seed) and a collection of **Scenes** and **Shots**.
*   **Scene**: A narrative unit derived from a screenplay.
*   **Shot**: An individual unit of video generation. It holds its own prompt, seed, and—crucially—**Timeline** (conditioning inputs).

### 4.2 Latent Handoff (The "Quantum Alignment" Fix)
To prevent "morphing" or inconsistency between chained clips, the system implements **Latent Handoff**.
*   Instead of just using the last frame of Video A as an image input for Video B, the system preserves the **Late-Stage Latents** from Video A.
*   These latents are passed into the LTX-2 pipeline for Video B, ensuring true temporal continuity even at the pixel generation level.
*   *Implementation*: See `worker.py` -> `generate_chained_video_task`.

### 4.3 Element Injection
The system allows defining **Elements** (Characters, Locations).
*   Users define an Element (e.g., "Hero" -> Reference Image).
*   Using `@Hero` in a prompt triggers the `ElementManager`.
*   The system injects proper visual conditioning (IP-Adapter images) or text substitutions automatically before generation.

### 4.4 The "Director" Prompting
The `StoryboardManager` doesn't just pass prompts blindly. It uses **Gemma 3** to act as a "Virtual Director".
*   It analyzes the previous shot's context.
*   It rewrites the user's simple instruction (e.g., "walks away") into a rich cinematic prompt (e.g., "Slow dolly out as the figure retreats into shadows...").

---

## 5. Detailed Module & File Documentation

### 5.1 Backend (`backend/`)

#### `server.py`
*   **Role**: API Gateway.
*   **Key Dependencies**: `FastAPI`, `BackgroundTasks`, `managers.*`.
*   **Key Routes**:
    *   `POST /generate/advanced`: The unified entry point for all generation. It routes requests to the `worker`.
    *   `GET /events`: SSE endpoint for pushing progress bars and logs to the client.
    *   `POST /projects/{id}/storyboard/commit`: Saves parsed screenplay structures to the DB.
*   **Design Note**: Uses `BackgroundTasks` to offload generation to `worker.py` without blocking the HTTP response.

#### `worker.py`
*   **Role**: The "Brain" of the operation. Executes AI pipelines.
*   **Key Classes**: `ModelManager` (Singleton for loading LTX-2/Flux models).
*   **Key Functions**:
    *   `generate_chained_video_task()`: Complex logic for handling "Smart Continue". It manages the loop of generating Chunk N, trimming overlap, and preparing conditioning for Chunk N+1.
    *   `generate_standard_video_task()`: Handles single-shot T2V or I2V.
*   **Complex Logic**: Contains the ffmpeg stitching/trimming logic (using `-filter_complex`) to merge generated chunks seamlessly.

#### `models/flux_wrapper.py`
*   **Role**: Bridge to `flux2` dependency.
*   **Logic**:
    *   Implements `FluxInpainter` class.
    *   Manages IP-Adapter loading (Visual Conditioning) and injection into the Flux inference loop.
    *   Handles platform-specific optimizations (Freezing Float32 for MPS devices).

#### `config.py`
*   **Role**: System Configuration & Path Injection.
*   **Crucial Function**: `setup_paths()` - Adds the `LTX-2/packages/...` and `flux2/src` paths to `sys.path` so other modules can import them.

#### `database.py`
*   **Role**: Schema Definition.
*   **Models**:
    *   `Project`: Root container.
    *   `Shot`: The atomic unit. Contains `params_json` (generation config) and `status`.
    *   `Element`: Stores trigger words and reference image paths.
    *   `Job`: Transient record of an async generation task, linked to a Shot.

#### `managers/element_manager.py`
*   **Role**: Visual Consistency abstraction.
*   **Logic**:
    *   `inject_elements_into_prompt()`: Scans strings for `@LikeThis`. Returns enhanced text + list of image paths.
    *   `generate_visual()`: Calls Flux to generate "Character Sheets" for new elements.

#### `storyboard/manager.py`
*   **Role**: Narrative Logic.
*   **Logic**:
    *   `prepare_next_chunk()`: Determines what the next chunk of video needs (e.g., extracting the last frame of the previous chunk).
    *   `commit_chunk()`: Updates internal state after successful generation.

### 5.2 Frontend (`web-app/`)

#### `stores/timelineStore.ts`
*   **Role**: The "Grand Central Station" of frontend state.
*   **Tech**: Zustand + Zundo (Undo/Redo) + Persist (LocalStorage).
*   **Key Actions**:
    *   `inpaintShot`: Orchestrates the flow of uploading a frame -> uploading a mask -> triggering backend job.
    *   `loadProject`: Maps backend `snake_case` JSON to frontend `camelCase` interfaces.
*   **Risk**: The file is very large (800+ lines). It mixes UI state (toasts) with Data state (Projects). Future refactor candidate.

#### `components/Inspector/`
*   **Role**: Properties panel for the selected shot.
*   **Files**:
    *   `InspectorPanel.tsx`: Main container.
    *   `ConditioningEditor.tsx`: Advanced UI for setting up Image/Video inputs for specific frames.

---

## 6. Data Flow & Control Flow

**Scenario: Generating a "Smart Continue" Video**

1.  **User Interaction** (Frontend): User selects a shot, clicks "Extend", and types a prompt.
2.  **State Update** (`timelineStore`): Store updates local shot status to "Queued".
3.  **API Call** (`server.py`): Frontend POSTs to `/generate/advanced`.
4.  **Job Creation** (`server.py`): Server creates a `Job` record in SQLite (`status="pending"`) and returns `job_id`.
5.  **Task Dispatch** (`server.py`): Server adds `generate_video_task` to Background Tasks.
6.  **Worker Execution** (`worker.py`):
    *   Retrieves project context.
    *   **Inference Loop**:
        *   Chunk 1: Generates video.
        *   **Latent Extraction**: Captures the latent tensor of the last frames.
        *   Chunk 2: Initializes LTX-2 with the captured latents (Latent Handoff).
    *   **Stitching**: Uses `ffmpeg` to concatenate chunks, removing overlap frames (approx 24 frames).
7.  **Notification** (`events.py`): Worker broadcasts "progress" and "complete" events via SSE.
8.  **Update** (`timelineStore`): Frontend receives SSE, updates Shot with new `videoUrl`, and refreshes the player.

---

## 7. Configuration & Environment

*   **Config File**: `backend/config.py` acts as the single source of truth.
    *   `PROJECTS_DIR`: Where user data lives.
    *   `LTX_DIR`, `FLUX_DIR`: Where model weights live.
*   **Environment Variables**:
    *   The system largely relies on hardcoded relative paths in `config.py` assuming standard installation structure.
    *   `SAM_SERVICE_PORT`: Defaults to specific port for SAM microservice.

---

## 8. External Integrations

*   **LTX-2 (Lightricks)**:
    *   Used for: Text-to-Video, Image-to-Video.
    *   Integration: Direct Python import of `ltx_pipelines`.
*   **Flux 2 (Black Forest Labs)**:
    *   Used for: Text-to-Image (Element generation), In-Painting.
    *   Integration: `backend/models/flux_wrapper.py`.
*   **SAM 3 (Meta)**:
    *   Used for: Segmentation / Auto-Masking.
    *   Integration: Microservice architecture. Backend talks to SAM via HTTP (`localhost:PORT/segment`).

---

## 9. Extension & Customization Guide

### How to add a new Model (e.g., a new Video Model)?
1.  **Backend**: Add the model loader in `worker.py` -> `ModelManager.load_pipeline`.
2.  **Config**: Add path constants in `config.py`.
3.  **API**: Update `ShotConfig` pydantic model in `server.py` to accept new parameters.
4.  **Frontend**: Update `Shot` interface in `timelineStore.ts` and add UI controls in `InspectorPanel.tsx`.

### Safe Modification Zones
*   **Frontend Components**: `web-app/src/components` are modular and safe to modify.
*   **Prompt Logic**: `managers/element_manager.py` logic for prompt injection is isolated.

### Danger Zones (Modify with Caution)
*   `worker.py` (Stitching Logic): The `ffmpeg` commands for trimming and fading are brittle and tuned for specific FPS/Samplerates.
*   `timelineStore.ts`: Modifying the `Project` structure here requires a matching migration in `server.py` (ProjectState model) and `database.py`.

---

## 10. Debugging & Maintenance

### Logging
*   **Server Log**: `backend/server_debug.log` contains detailed request/response and error traces.
*   **Frontend Log**: Browser Console.

### Common Failure Points
1.  **VRAM OOM**: Running Flux and LTX-2 simultaneously. The `ModelManager` attempts to unload pipelines, but fragmentation can occur.
    *   *Fix*: Restart backend.
2.  **MPS bfloat16 Crash**: `unsupported input/output datatypes to MPSNDArrayMatrixMultiplication kernel`
    *   *Cause*: Gemma text encoder or LoRA fusion using `bfloat16` which MPS doesn't support.
    *   *Fix*: Codebase uses `torch.backends.mps.is_available()` to force `float16`.
3.  **Black Frames (MPS)**: On Mac, FP16 VAE can cause black output.
    *   *Fix*: `worker.py` forces Float32 for VAE on MPS devices.
4.  **FFmpeg Stitching**: Audio/Video desync in "Smart Continue".
    *   *Debug*: Check `server_debug.log` for "Stitch failed" messages.
5.  **MPS VAE Decode OOM**: `MPS backend out of memory` during video encoding.
    *   *Cause*: Transformers not unloaded + default tiling too large for unified memory.
    *   *Fix*: System now uses `TilingConfig.mps_default()` (256px, 32 frames) and calls `clear_cached_models()` before VAE decode.

---

## 11. Known Risks & Technical Debt

1.  **Concurrency Limitations**: The system is designed as **Single-Tenant**. The `ModelManager` uses a global lock (`asyncio.Lock`). Multiple users trying to generate simultaneously will be queued, but race conditions on VRAM unload/load may occur.
2.  **Large Store File**: `timelineStore.ts` is becoming a "God Object". It should be split into slices (ProjectSlice, UIStateSlice).
3.  **Hardcoded Paths**: Relative paths depend on specific folder structures (`../LTX-2`). This makes containerization (Docker) tricky without refactoring `config.py`.
4.  **Database Migration**: No Alembic/Migration tool is currently configured. Schema changes require manual DB resets or careful SQL manipulation.
5.  **Validation**: Frontend and Backend share data structures but define them separately (TypeScript Interfaces vs Python Pydantic Models). Start of drift risk.

---

## 12. Glossary
*   **Latent Handoff**: Passing internal model state (latents) from the end of one generation to the start of the next.
*   **Shot**: A single video clip generation unit.
*   **Element**: A reusable entity (Character/Location) with visual consistency rules.
*   **Smart Continue**: The feature that orchestrates Latent Handoffs to extend video indefinitely.
