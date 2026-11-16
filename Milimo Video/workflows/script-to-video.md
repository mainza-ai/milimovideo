---
type: workflows
created: 2026-10-09
updated: 2026-10-09
tags: storyboard script-to-video pre-viz generation
sources:
  - README.md
  - docs/03_ai_pipelines.md
  - docs/05_execution_flow.md
  - docs/user_guide.md
related:
  - ../systems/storyboard-engine.md
  - ../concepts/story-elements.md
  - ../pipelines/chained-generation.md
  - ../systems/sse-eventing.md
---

# Script to Video

## Goal

Turn a written screenplay into generated video shots assembled on the timeline —
the flagship end-to-end flow.

## Prerequisites

- Studio running (backend :8000, frontend :5173; SAM service optional for this flow)
- A project with Story Elements defined (characters/locations with `@triggers` and
  reference images) for consistent identities

## Steps

1. **Paste the script** — open the Storyboard view, paste screenplay text into
   `ScriptInput`.
2. **Parse** — click parse (regex `ScriptParser`) or the Brain icon (Gemma 3 AI
   parse → structured scenes/shots JSON). Review the parsed scenes in
   `StoryboardView`.
3. **Match elements** — the `ElementMatcher` auto-links shots to your elements
   (8-signal scoring); re-run `/match-elements` from the UI if you edit the script.
4. **Commit** — `commitStoryboard` writes Scene and Shot records to the DB.
5. **Pre-viz** — `Generate Thumbnails`: Flux 2 renders 512×320 concept art per shot
   (`is_thumbnail` jobs; SSE matches by `shot_id`). This is also your image-to-video
   seed — thumbnails become frame-0 conditioning.
6. **Generate shots** — per shot (or batch): `POST /storyboard/shots/{shot_id}/generate`.
   The manager enriches the prompt (previous-shot continuity + element triggers +
   narrative context), enhances it via the LLM, selects the pipeline (ti2vid /
   ic_lora / keyframe / chained for >505 frames), and streams SSE progress.
7. **Assemble** — `pushStoryboardToTimeline` (or generate directly onto shots);
   arrange on the magnetic V1 track, trim, add overlays on V2 and audio on A1.
8. **Save** — `Cmd+S` or the 2s-debounce autosave (`PUT /projects/{id}/save`).

## Tips

- Define elements **before** parsing — the matcher only links what exists.
- Concept-art thumbnails double as image-to-video conditioning (priority: manual
  image > thumbnail > previous shot's last frame).
- Shots over 505 frames automatically take the chained path (Quantum Alignment) —
  expect per-chunk SSE progress, not one smooth bar.
- Batch-generate overnight; the MemoryManager unloads/reloads models between video
  and image phases automatically.

## Related

- [[../systems/storyboard-engine.md]] — the machinery
- [[../concepts/story-elements.md]] — the consistency mechanism
- [[../pipelines/chained-generation.md]] — long-shot generation
- [[../systems/sse-eventing.md]] — progress and completion events
