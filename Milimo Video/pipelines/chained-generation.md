---
type: pipelines
created: 2026-10-09
updated: 2026-10-09
tags: chained-generation quantum-alignment ltx-2 long-video
sources:
  - docs/03_ai_pipelines.md
  - docs/user_guide.md
  - docs/ltx2-bible.md
related:
  - ../models/ltx-2.md
  - ../concepts/memory-slots.md
  - ../systems/storyboard-engine.md
  - ../decisions/cloud-engines.md
---

# Chained Generation (Quantum Alignment)

## Trigger Conditions

**Current code (verified 2026-11-16):** when `pipeline_type == "ti2vid"` and
`num_frames > config.NATIVE_MAX_FRAMES` (**121**), `tasks/video.py` hands off to
`generate_chained_video_task()` in `tasks/chained.py`. `StoryboardManager.chunk_size
= 121`, `overlap_frames = 25`. The older 505-frame/24-px latent-handoff description
in the docs is historical; the implemented path uses a 25-frame **pixel**
`media_sequence`, not a captured latent tail.

## Stages

1. `StoryboardManager.get_total_chunks()` computes the chunk count from
   `chunk_size=121`, `overlap=25`, `effective_step=96`.
2. For chunk 0: run the TI2Vid pipeline for 121 frames.
3. For chunk N>0: async-extract the last 25 frames (ffmpeg subprocess, glob in
   `workspace/storyboard_artifacts/`), pass them as `media_sequence`, generate 121.
4. Trim the first 25 frames from the decoded video in Python; trim audio by the
   same count (currently trimmed at the **end** — see audit).
5. Commit the chunk; repeat. Final stitch re-encodes with an ffmpeg `concat`
   filtergraph (`libx264`, CRF 18, AAC 192k), not `-c copy`.
6. Cancellation is checked between chunks and in the step callback; the
   pre-chunk `break` path can still stitch a partial film (see audit).

## Key Parameters

| Parameter | Value |
|---|---|
| `chunk_size` | 505 frames (LTX-2 native limit) |
| `overlap_frames` | 24 px requested overlap |
| `effective_step` | 481 frames of new content per chunk |

**Quantum Alignment math** (latent grid = 8 px per latent):

- `latent_slice_count = ceil((overlap_px − 1) / 8) + 1` → 4 latents for 24 px
- `effective_overlap_px = (latent_slice_count − 1) × 8 + 1` → 25 px
- `frames_to_trim = effective_overlap_px` → 25 frames trimmed per handoff
- `conditioning_latent = last_latent[:, :, start_slice:, :, :]` — the sliced latent
  seeds the next chunk

## Failure Modes

- **Misaligned overlap → "frozen anchor" artifacts**: if the overlap isn't snapped
  to the 8-pixel latent grid, the handoff region renders static. The math above
  exists specifically to prevent this.
- **Event-loop deadlock**: last-frame extraction must run ffmpeg in a subprocess
  (`asyncio.create_subprocess_exec`), never inline.
- **Memory**: each chunk is a full generation; the MemoryManager slot discipline
  (video slot held across chunks) keeps Flux 2 from being loaded mid-chain.

## Refinements (from the LTX-2 bible)

- **Product name**: the chained path is Milimo's **"Smart Continue"** feature.
- **Delegation precondition**: chained generation triggers only when
  `pipeline_type == "ti2vid"` **and** `num_frames > 505` — the other pipelines
  don't chain.
- **Latent handoff, precisely**: after each chunk's Stage 1 denoising, the full
  latent tensor is captured (`last_chunk_latent`); the conditioning slice is
  `last_chunk_latent[:, :, start_slice:, :, :]` — the *tail* of the previous
  chunk's latent — injected as a conditioning input seeding the next chunk.
- **Last-frame extraction**: `ffmpeg -sseof -0.1` grabs the final frame as frame-0
  conditioning for chunk N+1.
- **Trim template**: `ffmpeg -i chunk.mp4 -ss <trim_seconds> -c copy
  trimmed_chunk.mp4` where `trim_seconds = frames_to_trim / fps` — no re-encode.
  Stitching uses ffmpeg concat (file-list or filter method).
- **Progress**: SSE broadcasts per-chunk progress (`chunk i/N`); cancellation
  checks `active_jobs[job_id]["cancelled"]` between chunks.
- **StoryboardManager dual modes**: Worker/Generation `(job_id, prompt, params,
  output_dir)` for chained runs vs Server/Prep `(output_dir)` for shot-based
  storyboard workflows; `cleanup()` is currently a stub.
- **Frozen anchor, precisely**: 1 latent = 8 pixels in the temporal dimension; if
  the overlap isn't snapped to that grid, the handoff region renders static. The
  alignment math exists to make the trim exact, not approximate.

## Audit Notes (2026-11-16)

- `NATIVE_MAX_FRAMES` is 121, so chained generation starts far earlier than the
  505-frame story the docs told. This is an MPS-safety decision, not the LTX limit.
- **Audio desync:** video drops the first `frames_to_trim` frames; audio drops the
  last `trim_samples` (`tasks/chained.py:343-350`). Each chunk shifts audio ~1s.
- **Cancellation can produce a partial "success":** `break` at the pre-chunk check
  (`:252`) falls through to stitch/probe/complete. Cancel must raise, not break.
- **No resume:** a crash mid-chain leaves `*_part_*.mp4` files and no checkpoint;
  the job is not resumable.
- **No auto-negative:** chained chunks append static/freeze negatives, but not the
  standard auto-negative list used by `generate_standard_video_task`.
- Stitching re-encodes every chunk; this is slower and can drift A/V versus the
  documented `-c copy` template.

## Related

- [[../systems/engineering-audit-2026-11.md]] — GEN-05/GEN-04 evidence
- [[../models/ltx-2.md]] — the pipeline being chained
- [[../concepts/memory-slots.md]] — why the video slot stays held
- [[../systems/storyboard-engine.md]] — `StoryboardManager` modes and shot prep
- [[../decisions/cloud-engines.md]] — the proposed cloud hook for chained chunks
