---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: storyboard script-parsing element-matching pre-viz
sources:
  - docs/03_ai_pipelines.md
  - docs/05_execution_flow.md
  - README.md
related:
  - data-models.md
  - frontend-state.md
  - prompt-enhancement.md
  - ../concepts/story-elements.md
  - ../workflows/script-to-video.md
---

# Storyboard Engine

## Purpose

Turns a pasted screenplay into a producible shot list: parse → match elements →
commit → generate. The bridge between writing and the generation pipelines.

## Components

| Component | Role |
|---|---|
| `services/script_parser.py` | Regex-based `ScriptParser`: detects `INT.`/`EXT.` scene headings, groups dialogue, extracts action → `List[ParsedScene]` |
| `services/ai_storyboard.py` | `ai_parse_script()` — Gemma 3 text encoder used as a chat model with a cinematic system prompt; outputs strict JSON (1–8 varied shots per scene; dialogue paired with close_up/medium); falls back to the regex parser when Gemma isn't loaded |
| `services/element_matcher.py` | `ElementMatcher` — deterministic 8-signal scoring engine linking shots to project elements (see [[../concepts/story-elements.md]]) |
| `storyboard/manager.py` | `StoryboardManager` — chunk prep for chained gen, last-frame extraction, shot-based prompt enrichment; runs in two modes: Worker/Generation `(job_id, prompt, params, output_dir)` and Server/Prep `(output_dir)` |
| `StoryboardView` / `ElementSlice` | Frontend: `parseScript`/`aiParseScript`, `commitStoryboard`, `generateThumbnails`, `pushStoryboardToTimeline` |

## Key Interactions

1. **Parse**: `POST /projects/{id}/script/parse` (regex) or
   `/projects/{id}/storyboard/ai-parse` (Gemma). The AI path strips markdown
   fences, regex-searches for a JSON array, then `_validate_scenes()` normalizes
   shot types (unknown → `"medium"`) and coerces dialogue/character fields.
2. **Match**: `match_elements()` runs after either parser (and is re-triggerable
   from the UI via `/storyboard/match-elements`). `ElementManager.inject_elements_into_prompt()`
   later injects the matched trigger words/images at generation time (the manager
   itself has no such method; the name in early docs was wrong).
3. **Commit**: `POST /projects/{id}/storyboard/commit` — smart-merges scenes/shots
   straight to the DB (no model involved), creating Scene and Shot records.
4. **Thumbnails (pre-viz)**: `POST /projects/{id}/storyboard/thumbnails`
   (shot_ids, 512×320, force) → per shot a Flux 2 job with `is_thumbnail=True`;
   the SSE `complete` carries `shot_id` and the store updates only `thumbnailUrl`.
5. **Generate**: `POST /shots/{shot_id}/generate` (or batch via
   `/projects/{id}/storyboard/batch-generate`) → `prepare_shot_generation()` loads
   the shot, grabs the previous shot's last frame
   as continuity conditioning, applies prompt priority (`shot.prompt` manual override
   beats `shot.action`), prefixes narrative context
   ("Following the previous shot where {prev.action}…"), injects element triggers,
   then dispatches the video task.

## Gotchas

- The AI parser's JSON extraction exists because Gemma wraps output in fences or
  prose — `_extract_json()` handles both, and raises `ValueError` if nothing parses.
- Shot `index` is 1-based within a scene; `prepare_shot_generation` finds the
  previous shot via `Shot.index - 1`.
- Last-frame extraction shells out to ffmpeg (`-sseof -0.1`) via
  `asyncio.create_subprocess_exec` — never inline, or the event loop deadlocks.
- Thumbnail SSE events match by `shot_id`, not `lastJobId` — the one place
  `ServerSlice` doesn't use the job id.

## Audit Notes (2026-11-16)

See [[engineering-audit-2026-11.md]] for evidence:

- **`prepare_shot_generation()` is buggy:** `replacement` is referenced after the
  element loop (`manager.py:345`), so an empty match list raises
  `UnboundLocalError`; a non-empty list appends a duplicate `. Setting: …` for every
  match. This corrupts prompts and can 500 the generate endpoint.
- **Seed is always randomized** for `generate_shot` (`storyboard.py:606-613`), so
  the inspector's seed control has no effect for storyboard shots.
- **Commit deletes orphan shots** by `(scene, index)` without archiving generated
  media — a shorter re-parse can silently orphan finished takes.
- **AI parse loads the 19B LTX pipeline inside the request** (`storyboard.py:217-221`),
  evicting Flux and blocking the event loop while Gemma runs.
- **Regex parse is stateful:** `ScriptParser._dialogue_counter` persists across
  calls, so parsing the same script twice can assign different shot types.
- **Element matching is substring-based:** trigger "hero" matches "superhero";
  cross-project fallback can pull an identity from another project.
- Endpoint truth: `script/parse`, `storyboard/ai-parse`, `shots/{id}/generate`.

## Related

- [[engineering-audit-2026-11.md]] — GEN-01/GEN-06/GEN-07 evidence
- [[data-models.md]] — Scene/Shot records and storyboard request schemas
- [[frontend-state.md]] — ElementSlice actions
- [[prompt-enhancement.md]] — the Gemma path reused for AI parsing
- [[../concepts/story-elements.md]] — what the matcher matches
- [[../workflows/script-to-video.md]] — the user-facing flow
