---
type: workflows
created: 2026-10-09
updated: 2026-11-17
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
- Open `http://localhost:5173/studio`
- A project with Story Elements defined (characters/locations with `@triggers` and
  reference images) for consistent identities

## Steps

1. **Paste the script** — open **Story → Script**, paste screenplay text.
2. **Parse** — **Parse & replace storyboard** runs the local `ScriptParser` and
   archives the current board first. **AI parse (queued)** sends the heavy parse
   through the GPU ledger and polls/commits the result.
3. **Match elements** — `ElementMatcher` auto-links shots to master references;
   re-run match from the script/board UI after edits.
4. **Board and edit** — the parsed scenes/shots appear in the Story spine, board,
   and shot list. Arrange meaning on the board; order remains `sequence_index`.
5. **Pre-viz** — **Generate thumbnails** renders concept art per shot. A preview
   still can be attached as start or end conditioning in Generate.
6. **Generate shots** — per shot or via **Queue pending** / the batch matrix:
   `POST /shots/{shot_id}/generate` (or the advanced route). The manager enriches
   the prompt, selects the pipeline (ti2vid / ic_lora / keyframe / chained above
   the native 121-frame chunk), and streams SSE progress.
7. **Assemble** — generate takes onto shots, then cut in Edit: magnetic V1, free
   V2/A1, trims, speed, fades, captions, keyframes, and render range.
8. **Save** — `⌘S` or the top-bar Save control (`PUT /projects/{id}`). Saves are
   serialized and revision-checked.
9. **Review and deliver** — comment/approve in Review, then export from Deliver
   with fit, encoding, captions, and range.

## Tips

- Define elements **before** parsing — the matcher only links what exists.
- Concept-art thumbnails double as image-to-video conditioning (priority: manual
  image > thumbnail > previous shot's last frame).
- Shots over the native 121-frame chunk take the chained path (Quantum Alignment)
  — expect per-chunk SSE progress, not one smooth bar.
- Batch-generate overnight; one heavy GPU job runs at a time through the durable
  queue. Pause/resume dispatch from Setup or Generate when needed.

## Related

- [[../systems/storyboard-engine.md]] — the machinery
- [[../concepts/story-elements.md]] — the consistency mechanism
- [[../pipelines/chained-generation.md]] — long-shot generation
- [[../systems/sse-eventing.md]] — progress and completion events
