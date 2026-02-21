---
name: milimo-system-architect
description: Provides deep expertise on the Milimo Video system architecture, including the split React/FastAPI design, Zustand God Store, Server-Sent Events (SSE) job tracking, and the SAM 3 microservice integration. Use this when you are adding new features that cross the frontend/backend boundary, modifying database schemas, or debugging state synchronization issues.
---

# Milimo System Architect Skill

As an expert on the Milimo Video system architecture, you understand how the React/Vite frontend communicates with the FastAPI backend to orchestrate the AI Cinematic Studio.

## Core Architectural Understanding

1. **Split Architecture**:
   - **Frontend**: React 18, Vite, TypeScript.
   - **Backend**: Python 3.10+, FastAPI, SQLModel (SQLite).
   - **Microservice**: SAM 3 runs on port `8001` as a separate, isolated backend service for object tracking and segmentation.

2. **The God Store Pattern (Frontend State)**:
   - The application relies on a massive Zustand store (`timelineStore.ts`) built from 7 slices: Project, Shot, Playback, UI, Track, Element, Server.
   - **CRITICAL**: Never add new React Contexts for global state. Always add to the appropriate Zustand slice.
   - Performance is maintained via `useShallow` selectors and transient updates (store `.getState()` references in `useRef` for render loops).
   - The timeline relies on a "Magnetic V1" layout: Track 0 (V1) items snap together sequentially. V2 and A1 tracks use free placement (`startFrame`).

3. **Job Management & SSE (Backend/Frontend Sync)**:
   - Long-running generation/inpainting tasks are offloaded to `BackgroundTasks` in FastAPI.
   - The backend `job_utils.py` tracks active jobs in a global mapping and broadcasts progress via Server-Sent Events (SSE).
   - The frontend `SSEProvider.tsx` listens for `progress`, `complete`, `error`, and `cancelled` events, syncing status directly to the Shot or corresponding Element.
   - On page load, `jobPoller.ts` triggers one-shot queries to `/status/{lastJobId}` to hydrate active state from the database.

4. **Multi-Track Playback Loop**:
   - Playback is driven by a headless `requestAnimationFrame` loop in `PlaybackEngine.tsx`.
   - Audio sync is handled by `GlobalAudioManager.ts` (Web Audio API) to bypass Safari media tag drift. Do not use `<audio>` elements for timeline tracks.
   - Video elements (`CinematicPlayer.tsx`) listen to the store via strict drift-correction thresholds (250ms).

## Guidelines for Feature Implementation
- **Adding an Endpoint**: Create it in the appropriate router domain (`projects.py`, `jobs.py`, `assets.py`, `elements.py`, `storyboard.py`). Wire it to `schemas.py` and `database.py`.
- **Handling Progress**: Any task that takes longer than 2 seconds MUST log a `Job` to the DB and broadcast progress via `job_utils.broadcast_progress`. Provide actionable progress messages.
- **Microservice Integration**: When calling SAM 3, do not import its modules. Make HTTP requests via `InpaintingManager` or `TrackingManager` to localhost:8001.

## Rules
- **No Canvas Rendering for NLE**: The timeline must remain DOM-based CSS rendering. Do not attempt to rewrite the timeline in HTML5 Canvas.
- **Maintain Single Source of Truth**: `timelineStore.ts` dictates the state. Local component state is only for transient UI interactions (e.g. dragging, hover states).
- **Graceful Cancellation**: All long-running tasks must periodically check `active_jobs[job_id]["cancelled"]` and `raise RuntimeError("Cancelled")` to free GPU resources.
