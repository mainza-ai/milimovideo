# Milimo Video — Execution Flow

## 1. Video Generation Sequence

```mermaid
sequenceDiagram
    participant UI as Frontend (React)
    participant Store as TimelineStore
    participant API as FastAPI Server
    participant DB as SQLite (SQLModel)
    participant Utils as job_utils
    participant Worker as BackgroundTask
    participant ME as ModelManager
    participant LTX as LTX-2 Pipeline
    participant FFmpeg as FFmpeg

    Note over UI: User clicks "Generate" in InspectorPanel
    
    UI->>Store: generateShot(shotId)
    Store->>Store: updateShot(id, {isGenerating: true})
    Store->>API: POST /generate/advanced {project_id, shot_config}

    API->>DB: Create Job (status=pending)
    API->>DB: Update Shot (status=generating, last_job_id)
    API->>Utils: Check GPU Semaphore Limit
    API->>Utils: Register active_jobs[job_id]
    API->>Worker: BackgroundTasks.add_task(queue_video_task)
    API-->>UI: Return 200 {job_id, status: "queued"}

    Note over UI: SSE connection already active

    Worker->>Utils: update_job_db(job_id, "processing")
    Worker->>Utils: broadcast_progress(job_id, 0, "Loading models...")
    Utils-->>UI: SSE event: {type: "progress", job_id, progress: 0}
    UI->>Store: handleServerEvent("progress", data)
    Store->>Store: updateShot(shotId, {progress: 0, statusMessage: "Loading models..."})

    Note over Worker: Pipeline Selection
    Worker->>Worker: Auto-detect pipeline type (ti2vid / ic_lora / keyframe)
    Worker->>Worker: Check num_frames == 1 → delegate to Flux 2 (skip LTX)
    Worker->>Worker: Check num_frames > 505 → delegate to chained gen

    Worker->>ME: load_pipeline("ti2vid", loras)
    ME->>ME: Unload previous pipeline (gc + cache clear)
    ME->>LTX: TI2VidTwoStagesPipeline(checkpoint, ...)
    ME->>ME: MPS fix: force VAE to float32
    ME-->>Worker: Pipeline ready

    Note over Worker: Prompt Enhancement
    Worker->>Worker: llm.enhance_prompt(prompt, text_encoder, image_path)
    Worker->>Utils: update_shot_db(shot_id, {enhanced_prompt_result: ...})
    Worker->>Utils: broadcast_progress(job_id, 5, enhanced_prompt)
    Utils-->>UI: SSE event: {type: "progress", enhanced_prompt: "..."}

    Note over Worker: Image Conditioning (if timeline items present)
    Worker->>Worker: Resolve image paths (URL decode, project path join)
    Worker->>Worker: Load & encode conditioning images via VAE

    Note over Worker: Run Pipeline
    Worker->>LTX: pipeline(prompt, images, num_frames, ...)
    
    loop Denoising Steps (8 distilled / 40 base)
        LTX->>LTX: Step N
        LTX-->>Worker: step_callback(step, total)
        Worker->>Utils: Check active_jobs[job_id].cancelled
        alt Cancelled
            Worker->>Utils: update_job_db(job_id, "cancelled")
            Worker-->>Utils: broadcast_progress(status="cancelled")
        end
        Worker->>Utils: broadcast_progress(job_id, progress%, eta)
        Utils-->>UI: SSE event (progress update)
    end

    LTX-->>Worker: Return video tensor

    Worker->>FFmpeg: encode_video(tensor, output_path) — libx264 + yuv420p + movflags faststart
    Worker->>FFmpeg: Asynchronous generate_thumbnail(video_path)
    Worker->>Utils: update_job_db(job_id, "completed", output_path, thumbnail_path)
    Worker->>Utils: update_shot_db(shot_id, {video_url, thumbnail_url, status: "completed"})
    Worker->>Utils: broadcast_progress(job_id, 100, "completed", video_url, thumbnail_url)
    
    Utils-->>UI: SSE event: {type: "complete", job_id, video_url, thumbnail_url}
    UI->>Store: handleServerEvent("complete", data)
    Store->>Store: updateShot(shotId, {isGenerating: false, videoUrl, thumbnailUrl, progress: 100})
    Store->>Store: triggerAssetRefresh()

    UI->>API: GET /projects/{id}/generated/{job_id}.mp4
    UI->>UI: CinematicPlayer loads video
```

## 2. Image Generation Sequence

```mermaid
sequenceDiagram
    participant UI as ImagesView
    participant API as FastAPI
    participant DB as SQLite
    participant Worker as BackgroundTask
    participant Flux as FluxInpainter

    UI->>API: POST /generate/image {project_id, prompt, reference_images, enable_ae, enable_true_cfg, ...}
    API->>DB: Create Job (type="generation")
    API->>Worker: BackgroundTasks.add_task(generate_image_task)
    API-->>UI: {job_id, status: "queued"}

    Note over Worker: Element Resolution
    Worker->>Worker: Merge element_images + reference_images
    Worker->>DB: Lookup UUID elements → resolve paths
    Worker->>DB: Scan all project elements for implicit trigger words in prompt
    Worker->>Worker: Append missing trigger words to prompt

    Note over Worker: Model Load (with AE hot-swap check)
    Worker->>Flux: load_model(enable_ae)
    Flux->>Flux: Check last_ae_enable_request — reload if changed
    Flux->>Flux: Load IP-Adapter if reference images present

    Worker->>Flux: generate_image(prompt, ip_adapter_images, enable_true_cfg, ...)
    
    loop Inference Steps (25)
        Flux-->>Worker: flux_callback(step, total)
        Worker->>Worker: Check active_jobs[job_id].cancelled
        Worker-->>UI: SSE progress update
    end

    Flux-->>Worker: PIL.Image result
    Worker->>Worker: Save as JPG (quality=95) + thumbnail (¼ size)
    Worker->>DB: Create Asset record (with meta_json: prompt, seed, guidance, refs)
    Worker->>DB: Update Job (completed)
    Worker-->>UI: SSE {type: "complete", url, asset_id, type: "image"}
```

## 3. Inpainting Sequence

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as FastAPI
    participant InpMgr as InpaintingManager
    participant SAM as SAM 3 :8001
    participant Flux as FluxInpainter

    Note over SAM: Sam3Processor + inst_interactive_predictor

    alt Point-Based Masking
        UI->>API: POST /edit/segment {image, points}
        API->>InpMgr: get_mask_from_sam(image_path, points)
        InpMgr->>SAM: POST /predict/mask (multipart: image + points JSON)
        Note over SAM: inst_interactive_predictor.predict()
    else Text-Based Masking
        UI->>API: POST /edit/segment-text (multipart: image + text + confidence)
        API->>SAM: Direct proxy: POST /segment/text
        Note over SAM: Sam3Processor.set_text_prompt()
    else Multi-Object Detection
        UI->>API: POST /edit/detect (multipart: image + text + confidence)
        API->>SAM: Direct proxy: POST /detect
        Note over SAM: Sam3Processor → masks + bboxes + scores
    end

    SAM-->>InpMgr: Mask data (PNG or JSON)
    InpMgr->>InpMgr: Save mask as {base}_mask_{uuid6}.png
    InpMgr-->>API: mask_path
    API-->>UI: {mask_path}

    UI->>API: POST /edit/inpaint {image_path, mask_path, prompt}
    API->>API: Create Job DB record (type="inpaint", status="pending")
    API->>API: Register active_jobs[job_id]
    API->>InpMgr: BackgroundTasks.add_task(process_inpaint, job_id, ...)
    API-->>UI: {status: "queued", job_id, mask_path}

    InpMgr->>Flux: inpaint(image_RGB, mask_L, prompt, guidance=2.0, enable_ae=True)
    
    Note over Flux: RePaint loop
    Flux->>Flux: Encode image → latents (x_orig)
    Flux->>Flux: Downscale mask to latent size
    loop Denoising Steps
        Flux->>Flux: pred = model(x, ctx, ...)
        Flux->>Flux: x_pred = x + (t_prev - t_curr) * pred
        Flux->>Flux: x_known = t_prev * noise + (1-t_prev) * x_orig
        Flux->>Flux: x = mask * x_pred + (1-mask) * x_known
        Flux-->>InpMgr: step_callback(step, total)
        InpMgr->>InpMgr: update_job_progress(job_id, progress)
        InpMgr-->>UI: SSE progress event
    end
    Flux->>Flux: Decode latents → image (CPU offload on MPS)
    
    Flux-->>InpMgr: PIL.Image result
    InpMgr->>InpMgr: Save to projects/{id}/generated/inpaint_{job_id}.jpg
    InpMgr->>InpMgr: Create Asset DB record
    InpMgr->>InpMgr: update_job_db(job_id, "completed", output_path, thumbnail_path)
    InpMgr-->>UI: SSE {type: "complete", job_id, thumbnail_url, type: "inpaint", asset_id}
```

## 4. Chained Generation Sequence

```mermaid
sequenceDiagram
    participant Worker as BackgroundTask
    participant SBMgr as StoryboardManager
    participant ME as ModelManager
    participant LTX as TI2VidTwoStagesPipeline
    participant FFmpeg as FFmpeg

    Note over Worker: Triggered when ti2vid + num_frames > 505

    Worker->>SBMgr: __init__(job_id, prompt, params, output_dir)
    Worker->>SBMgr: get_total_chunks() → N chunks

    loop For each chunk i = 0..N-1
        Worker->>SBMgr: prepare_next_chunk(i, last_output)
        
        alt Chunk 0
            SBMgr-->>Worker: {prompt, images: user_conditioning}
        else Chunk > 0
            SBMgr->>FFmpeg: Extract last frame (asyncio.create_subprocess_exec ffmpeg)
            SBMgr-->>Worker: {prompt, images: [(last_frame, 0, 1.0)]}
        end

        Worker->>ME: load_pipeline("ti2vid")
        Worker->>LTX: pipeline(prompt, images, 505 frames)
        LTX-->>Worker: Video tensor + Stage 1 latent

        Note over Worker: Quantum Alignment
        Worker->>Worker: latent_slice_count = ceil((24-1)/8) + 1 = 4
        Worker->>Worker: effective_overlap = (4-1)*8 + 1 = 25 px
        Worker->>Worker: Slice last 4 latents from tensor
        Worker->>Worker: Store as conditioning for next chunk

        alt Chunk > 0
            Worker->>FFmpeg: Trim first 25 frames (ffmpeg -ss)
        end

        Worker->>SBMgr: commit_chunk(i, path, prompt)
        Worker->>Worker: Check active_jobs[job_id].cancelled
        Worker->>Worker: Broadcast progress (chunk i/N)
    end

    Worker->>FFmpeg: Concat all trimmed chunks → final.mp4
    Worker->>Worker: Update Job + Shot DB records
    Worker-->>UI: SSE complete event
```

## 5. Startup Sequence

```mermaid
sequenceDiagram
    participant User
    participant Script as run_backend.sh
    participant API as FastAPI (Uvicorn)
    participant DB as SQLite
    participant React as Vite Dev Server
    participant Store as TimelineStore

    User->>Script: ./run_backend.sh
    Script->>API: Start Uvicorn (port 8000)
    
    API->>API: Configure Logging
    API->>API: Create directories (projects/, uploads/, generated/)
    API->>DB: init_db() — SQLModel.metadata.create_all()
    
    Note over API: Lifespan: Startup
    API->>API: Capture event loop → job_utils.global_loop
    API->>DB: Find zombie jobs (status="processing")
    API->>DB: Mark zombies as failed (prevent stale locks)
    
    API->>API: Mount Static Files (/projects, /uploads, /generated)
    API->>API: Register API Router (5 sub-routers)
    API->>API: Register /events SSE endpoint

    User->>Script: ./run_frontend.sh
    Script->>React: npm run dev (port 5173)

    User->>React: Open localhost:5173
    React->>Store: Hydrate from localStorage (persist middleware)
    Store->>Store: Check getLastProjectId()
    
    alt Has Last Project
        Store->>API: GET /projects/{id}
        API->>DB: Query Project + Shots + Scenes
        API-->>Store: ProjectState JSON
        Store->>Store: setProject(project)
        
        Note over Store: Recover in-flight jobs
        loop For each shot with isGenerating
            Store->>API: GET /status/{lastJobId}
            API->>DB: Query Job status
            API-->>Store: Job status data
            Store->>Store: Sync shot state
        end
    end

    React->>API: GET /events (SSE subscribe)
    API->>API: EventManager adds client queue
```

## 6. Storyboard Flow

```mermaid
sequenceDiagram
    participant UI as StoryboardView
    participant Store as TimelineStore
    participant API as FastAPI
    participant Parser as ScriptParser
    participant DB as SQLite
    participant SBMgr as StoryboardManager
    participant ElMgr as ElementManager

    Note over UI: User writes screenplay text
    UI->>Store: parseScript(scriptText)
    Store->>API: POST /projects/{id}/storyboard/parse {script_text}
    API->>Parser: script_parser.parse_script(text)
    
    Note over Parser: Regex-based parsing
    Parser->>Parser: Detect scene headings (INT./EXT.)
    Parser->>Parser: Group dialogue (ALL CAPS → dialogue lines)
    Parser->>Parser: Extract action descriptions
    Parser-->>API: List[ParsedScene]
    API-->>Store: Parsed scenes with shots
    Store-->>UI: Display parsed storyboard

    Note over UI: User reviews and edits
    
    UI->>Store: commitStoryboard(editedScenes)
    Store->>API: POST /projects/{id}/storyboard/commit {scenes}
    
    API->>DB: Create/Update Scene records
    API->>DB: Create Shot records per scene
    API->>DB: Link shots to scenes and project
    API-->>Store: Updated project state
    
    Note over UI: User triggers generation per shot
    UI->>API: POST /storyboard/shots/{shot_id}/generate
    API->>SBMgr: prepare_shot_generation(shot_id, session)
    SBMgr->>DB: Load Shot → find prev shot in same scene
    SBMgr->>SBMgr: Extract last frame from prev shot (ffmpeg)
    SBMgr->>ElMgr: inject_elements_into_prompt(shot.prompt or shot.action, project_id)
    SBMgr->>SBMgr: Add narrative context prefix
    SBMgr-->>API: {prompt, images, element_images}
    API->>API: Create Job → BackgroundTask(generate_video_task)
```

### 6.1 AI Script Analysis Flow

```mermaid
sequenceDiagram
    participant UI as ScriptInput (Brain icon)
    participant Store as TimelineStore
    participant API as FastAPI
    participant ME as ModelManager
    participant Gemma as Gemma 3 Text Encoder
    participant AIS as ai_storyboard.py
    participant Parser as ScriptParser (fallback)

    UI->>Store: aiParseScript(scriptText)
    Store->>API: POST /projects/{id}/storyboard/ai-parse {script_text}

    API->>ME: manager.current_pipeline (check Gemma loaded)
    
    alt Gemma Available
        API->>AIS: ai_parse_script(text, text_encoder)
        AIS->>AIS: Build chat messages (system + user)
        AIS->>Gemma: text_encoder._enhance(messages, max_new_tokens=1024)
        Gemma-->>AIS: Raw JSON text
        AIS->>AIS: _extract_json() — strip fences, parse JSON
        AIS->>AIS: _validate_scenes() — normalize shot_types
        AIS-->>API: List[ParsedScene]
    else Gemma Not Loaded
        API->>Parser: script_parser.parse_script(text) — regex fallback
        Parser-->>API: List[ParsedScene]
    end
    
    API-->>Store: Parsed scenes with shots
    Store-->>UI: Display AI-analyzed storyboard
```

### 6.2 Concept Art Thumbnail Flow

```mermaid
sequenceDiagram
    participant UI as StoryboardSceneGroup (ImageIcon)
    participant Store as TimelineStore
    participant API as FastAPI
    participant DB as SQLite
    participant Worker as BackgroundTask
    participant Flux as FluxInpainter

    UI->>Store: generateThumbnails(shotIds)
    Store->>API: POST /projects/{id}/storyboard/thumbnails {shot_ids, width=512, height=320}

    loop For each shot without thumbnail_url
        API->>DB: Create Job (is_thumbnail=True)
        API->>DB: Update Shot (status=generating, last_job_id)
        API->>Worker: BackgroundTasks.add_task(generate_image_task)
    end
    API-->>Store: {queued: N, skipped: M}

    loop Per thumbnail job
        Worker->>Flux: generate_image(action_prompt, 512×320)
        Flux-->>Worker: PIL.Image
        Worker->>Worker: Save JPG + thumbnail
        Worker->>DB: Update Shot.thumbnail_url (is_thumbnail path)
        Worker->>DB: Update Job (completed)
        Worker-->>UI: SSE {type: "complete", shot_id, thumbnail_url, is_thumbnail: true}
    end

    Note over UI: ServerSlice matches by shot_id (not lastJobId)
    UI->>Store: updateShot(shotId, {thumbnailUrl}) — no generation state change
```

## 7. Project Save/Load Flow

```mermaid
flowchart TD
    subgraph "Save Flow"
        S1[User action triggers auto-save] --> S2[saveProject in ProjectSlice]
        S2 --> S3[PUT /projects/id/save]
        S3 --> S4[Upsert Project in DB]
        S3 --> S5[Sync Shots — Create/Update/Delete]
        S3 --> S6[Sync Scenes — Create/Update/Delete]
        S5 --> S7[Return updated ProjectState]
    end

    subgraph "Load Flow"
        L1[loadProject in ProjectSlice] --> L2[GET /projects/id]
        L2 --> L3[Query Project + Shots + Scenes from DB]
        L3 --> L4[Map camelCase for frontend]
        L4 --> L5[setProject in store]
        L5 --> L6[fetchElements for project]
        L5 --> L7[Save project_id to localStorage]
    end

    subgraph "Auto-Save Hook"
        AS1[useAutoSave in VisualTimeline] --> AS2{Project changed?}
        AS2 -->|Yes| AS3[Debounce 2 seconds]
        AS3 --> AS4[Call saveProject]
        AS2 -->|No| AS5[Skip]
    end
```

## 8. Cancellation Flow

```mermaid
sequenceDiagram
    participant UI as InspectorPanel
    participant API as FastAPI
    participant Utils as job_utils
    participant Worker as BackgroundTask

    UI->>API: POST /jobs/{job_id}/cancel
    API->>Utils: active_jobs[job_id]["cancelled"] = True
    API->>Utils: active_jobs[job_id]["status"] = "cancelling"
    API-->>UI: {status: "cancelling"}

    Note over Worker: Next denoising step callback
    Worker->>Utils: Check active_jobs[job_id]["cancelled"]
    Worker->>Worker: Raise RuntimeError("Cancelled by user")
    Worker->>Utils: update_job_db(job_id, "cancelled")
    Worker->>Utils: update_shot_db(shot_id, {status: "pending"})
    Worker-->>UI: SSE {status: "cancelled"}

    Note over API: Shutdown Flow
    API->>Utils: cancel_all_jobs() — flags all active jobs
    API->>API: await asyncio.sleep(2.0) — grace period
```

## 9. Playback Engine Loop

```mermaid
flowchart TD
    RAF[requestAnimationFrame Loop] --> GET[getState from store]
    GET --> PLAYING{isPlaying?}
    
    PLAYING -->|Yes| DT[Calculate delta time]
    DT --> ADVANCE["nextTime = currentTime + dt"]
    ADVANCE --> CACHED["Use cached layout from useRef"]
    CACHED --> MAXDUR["Get maxDuration (precomputed)"]
    MAXDUR --> END_CHECK{"nextTime >= maxDuration?"}
    END_CHECK -->|Yes| STOP[setIsPlaying false]
    END_CHECK -->|No| UPDATE[setCurrentTime nextTime]
    
    PLAYING -->|No| RESET[Reset lastTimeRef]

    UPDATE --> AUDIO["GlobalAudioManager.tick (Web Audio API)"]
    RESET --> AUDIO
    AUDIO --> MUTE{Track muted?}
    MUTE -->|Yes| VOL0["GainNode.gain = 0"]
    MUTE -->|No| VOL1["GainNode.gain = 1"]
    VOL0 --> SYNC["Drift check (0.3s tolerance)"]
    VOL1 --> SYNC
    SYNC --> RESYNC{"Drift > 0.3s?"}
    RESYNC -->|Yes| RESTART["Stop + restart AudioBufferSourceNode at offset"]
    RESYNC -->|No| NOP[Continue playback]
    RESTART --> RAF
    NOP --> RAF
    STOP --> RAF
```

> **Note:** `computeTimelineLayout` is cached in a `useRef` and only recomputed when `project.shots` changes, not every animation frame. Audio uses `AudioContext` + `AudioBufferSourceNode` (Web Audio API) instead of `HTMLAudioElement` for reliable Safari playback.
