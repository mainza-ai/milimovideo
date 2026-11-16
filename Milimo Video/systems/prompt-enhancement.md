---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: llm prompt-enhancement gemma ollama
sources:
  - docs/03_ai_pipelines.md
  - docs/ltx2-bible.md
related:
  - storyboard-engine.md
  - ../concepts/memory-slots.md
  - ../models/ltx-2.md
---

# Prompt Enhancement

## Purpose

Every generation prompt passes through `llm.py`'s `enhance_prompt()` before it
reaches a model — rewritten for cinematic quality by a local LLM, never a cloud API.

## Components

| Component | Role |
|---|---|
| `backend/llm.py` | Single dispatcher: `enhance_prompt(prompt, system_prompt, is_video, text_encoder)`; selects provider per config |
| **Gemma provider** (built-in) | Reuses the LTX-2 pipeline's Gemma 3 text encoder via `text_encoder._enhance()` — on-device, no extra VRAM |
| **Ollama provider** (local) | `POST /api/generate` to `localhost:11434`; any model (e.g. gemma3-27b); vision models detected via `/api/tags` (mllama/clip/llava families) and badged 👁️ in the frontend |
| `VIDEO_SYSTEM_PROMPT` / `IMAGE_SYSTEM_PROMPT` | Concise action-focused vs vivid detailed system prompts |
| `tasks/video.py::generate_enhanced_prompt()` | Video-path entry: reuses the already-loaded Gemma encoder; captures a visual description when an input image exists |

## Key Interactions

- **Callers:** `tasks/video.py` (ti2vid/ic_lora/keyframe/single-frame), `tasks/chained.py`
  (chunk 0 + narrative continuation for later chunks), `tasks/image.py`,
  `storyboard/manager.py`, and `services/ai_storyboard.py` (script parsing — same
  Gemma chat interface, different system prompt).
- **Ollama VRAM discipline:** every request sends `keep_alive: 0`
  (`MILIMO_OLLAMA_KEEP_ALIVE`, or the frontend "Keep Model Loaded" toggle), then an
explicit unload request fires after enhancement — a 27B model (~49GB) must not
linger into the generation phase. See [[../concepts/memory-slots.md]].
- **VLM structured extraction:** when an input image is present, the VLM returns JSON
  with `enhanced_prompt` and `is_reference_only`. If the image is a character sheet
  or turnaround (`is_reference_only: true`), the dispatcher **drops image
  conditioning entirely** and forces LTX-2 into pure text-to-video — frozen 2D
  reference art would otherwise break generation.
- Settings surface: `GET/PATCH /settings/llm`, `GET /settings/llm/models`;
  frontend `LLMSettings.tsx` (provider select, model dropdown, keep_alive toggle).

## Gotchas

- The Gemma provider is free only when the LTX-2 pipeline is already loaded; using
  it standalone still loads the text encoder into the video memory slot.
- `is_reference_only` is a contract between the VLM output and the video task — if
  the JSON parse fails, the fallback path may attempt image conditioning on a
  character sheet and produce a frozen result.
- Prompt enhancement runs before pipeline selection, so its SSE `progress` update
  (with `enhanced_prompt`) arrives at ~5% — before any denoising.

## Related

- [[storyboard-engine.md]] — AI script parsing reuses this path
- [[../concepts/memory-slots.md]] — why the LLM unloads immediately
- [[../models/ltx-2.md]] — the text encoder doing double duty
