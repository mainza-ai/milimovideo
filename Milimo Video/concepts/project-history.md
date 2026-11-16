---
type: concepts
created: 2026-10-09
updated: 2026-10-09
tags: history dev-diary milestones build-log
sources:
  - dev_diary/
related:
  - ../overview.md
---

# Project History

## Definition

A distilled timeline of the Milimo Video build, compiled from the `dev_diary/`
build log (Jan–Feb 2026): social updates, blog posts, and an explainer-video brief.
The diary's own rules: never pushed to git, entries under 300 words, written in a
confident build-in-public voice — technical deep-dives with wry humor and cinematic
metaphors ("ghost conditioning," "the 161GB problem"). Facts below are distilled;
interpretive color is the diary's, not the wiki's.

## Timeline

| When | Milestone |
|---|---|
| Jan 2026 (early) | Prompt-bias bug fixed (every generation ended "…and a model is posing"); controls simplified to Video-or-Image; progress bar + shot deletion; cross-scene consistency named as the next hard problem |
| Jan 2026 | LTX-2 pushed hard: 4K (3840×2160), 50/60fps projects, 20-second single-pass shots; persistence migrated JSON → SQLite with atomic saves; "Smart Continue" autoregression improved with deep conditioning |
| Jan 2026 | "Static pause" seam in chained video fixed via aggressive conditioning decay; all Director Mode prompt logic consolidated into StoryboardManager |
| Jan 2026 | Seamless infinite chaining cracked: "Quantum Latent Alignment" (overlap computed in latents, 1 latent frame = 8 pixel frames, exact pixel trim); static-image freeze bug fixed; frame-accurate durations, instant thumbnails |
| 2026-01-30 | Bug-bash: LTX-2 tuple unpacking, `encode_video` arg order, project deletion, legacy `/generated/` → project-scoped `/projects/{id}/` URLs; drag-drop thumbnails; last-project persistence |
| Jan/Feb 2026 | "Feature Ready": Storyboard Engine (script→scenes→shots), Element Manager with `@trigger` injection, Flux 2 in-painting, MPS black-image fix (float32), live UI feedback; SAM 3 deployed as microservice |
| 2026-02-01 | Visual Conditioning: IP-Adapter in the Flux pipeline with "Visuals Only" conditioning and prompt isolation (`@Leona` no longer pollutes prompts); ghost-conditioning and 97%-progress-hang fixes |
| 2026-02-02 | Multi-reference visual conditioning working on MPS: native `load_ae()` autoencoder, URL-vs-path fix, 768px / max-3-ref caps with cache clears |
| 2026-02-03 | Backend monolith → modular refactor (`server.py`/`worker.py` → `routes/` + `tasks/`) with a line-by-line parity audit |
| 2026-02-04 | UX polish: Smart Merge persistence, assets surviving reload, timeline autosave, SSE progress push, rewritten V1 playback alignment |
| 2026-02-09 | SAM 3 ported to Apple Silicon MPS (empty-tensor guards, `pin_memory` fixes, MATH SDPA backend, bfloat16→float casts); blog deep-dive on the SAM 3 integration |
| 2026-02-17 | "The 161GB problem" solved: `MemoryManager` slot system evicts heavyweights on demand (LTX 76GB + Flux 36GB + LLM 49GB vs 128GB RAM); vision-model 👁️ detection in settings |
| 2026-02-17 | **Acquisition announced**: Milimo Video acquired; terms stated as non-negotiable — stays open source, local-first, not a paid SaaS; team ramping toward the next major release |

Also in the log: a manifesto post ("Hollywood's Next Studio Is Open Source"),
an engineering essay on orchestration-over-generation, and a YouTube explainer
brief covering consistent identities, the storyboard engine, SAM 3 inpainting, and
"quantum chaining."

## Key Milestones

1. **Quantum Latent Alignment** — the signature breakthrough enabling seamless
   infinite video chaining.
2. **SAM 3 integration** — text/click segmentation + object tracking turned the
   generator into an editing suite; required a major CUDA→MPS port.
3. **Storyboard Engine + Element Manager** — script parsing plus persistent
   `@trigger` character identity: the answer to AI filmmaking's consistency problem.
4. **Flux 2 in-painting with IP-Adapter conditioning** — masked editing with prompt
   isolation, working on MPS.
5. **MPS-first stability campaign** — black-frame fixes, OOM mitigations, and the
   MemoryManager slot system.
6. **Architecture hardening** — SQLite migration, modular backend refactor with
   parity audit, SSE updates, autosave.
7. **Acquisition (2026-02-17)** — with open-source continuation as the stated
   non-negotiable term.

## Open Threads (stated as plans, not shipped, at time of writing)

- Cross-scene narrative consistency ("teaching the machine to remember the last scene")
- Advanced video controls: camera motion, keyframing
- Vision-model-conditioned prompting (models reading conditioning frames)
- Post-acquisition next major release

## Related

- [[../overview.md]] — what the project became
