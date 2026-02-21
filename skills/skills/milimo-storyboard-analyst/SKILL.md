---
name: milimo-storyboard-analyst
description: Expertise in the Milimo Video Storyboard pipeline, from script parsing (Regex vs AI via Gemma) to generating concept art thumbnails and handling the Smart Element Matching engine. Use this when debugging storyboard extraction, prompt generation for chained video chunks, or modifying the scene/shot hierarchy logic.
---

# Milimo Storyboard Analyst Skill

As the Milimo Storyboard Analyst, your domain is transforming plain text screenplays into generation-ready, strictly formatted data structures (`Scene` and `Shot` records), and enriching those structures with intelligent context.

## 1. Script Parsing Pipelines

The frontend `StoryboardView.tsx` accepts raw script text. The backend processes it through two main parsing methodologies:

### A. The Regex Parser (`services/script_parser.py`)
- Fast, deterministic. Good for perfectly formatted standard screenplays.
- Uses regex to detect `INT.`/`EXT.` (Scenes), ALL CAPS (Character Names), and action blocks.
- **Failures**: Will miss non-standard formatting, prose descriptions, or poorly formatted text.

### B. The AI Parser (`services/ai_storyboard.py`)
- Dispatched via `POST /storyboard/ai-parse` when the brain icon is clicked.
- Routes through the LTX-2 Text Encoder's chat completion interface (`_enhance()`), defaulting to Gemma 3.
- Instructs the AI (via `AI_STORYBOARD_SYSTEM_PROMPT`) to act as a storyboard artist and build a cinematic `[ { "scene_heading": "...", "shots": [ ... ] } ]` JSON array.
- Evaluates implicit action descriptions to generate varied, appropriate cinematic `shot_types` (`close_up`, `wide`, `tracking`, etc.).
- **Fallback**: If Gemma unavailable, automatically routes back to Regex parser.

## 2. Smart Element Matching (`services/element_matcher.py`)

After a script is parsed but before it is committed to the database, the backend attempts to auto-link the newly discovered shots to existing Project Elements (`characters`, `locations`, `items`).

- **No LLM required**: Evaluates 8 discrete signals deterministically.
- Calculates a composite confidence score:
  - Exact character match: `1.0`
  - Trigger word in action: `0.95`
  - Name in action: `0.85`, etc.
- Matches with score `>= 0.35` are linked into the `shot.matched_elements` JSON field.
- **Why it matters**: `StoryboardManager` uses this data to inject visual conditioning (IP-Adapter reference images) into the generation pipeline for that shot.

## 3. Thumbnail Generation & The Job Queue

- UI triggers thumbnail generation: `POST /projects/{id}/storyboard/thumbnails`.
- Generates 512x320 concept art using Flux 2 (`generate_image_task`).
- Creates a backend `Job` marked with `is_thumbnail=True`.
- The `BackgroundTasks` worker fulfills the generation, saves to `Shot.thumbnail_url`, and fires an SSE `"complete"` event containing `shot_id` instead of `lastJobId`.
- **CRITICAL**: The frontend ServerSlice deliberately ignores `thumbnailUrl` updates if they do not match `shot.lastJobId` unless `is_thumbnail: true` is set, ensuring video generation jobs and thumbnail generation jobs do not conflict in the UI state.

## 4. Continuity (The Pipeline Handoff)

- To ensure flow across scenes, when `StoryboardManager.prepare_shot_generation()` is called on shot `N`, it attempts to pull the last frame of shot `N-1`.
- Uses `asyncio.create_subprocess_exec` ffmpeg extraction (`-sseof -0.1`) to grab the frame *without* blocking the FastAPI event loop.
- Modifies the generation request to include this extracted image as `conditioning_image` at `frame_0`.
