---
type: decisions
created: 2026-10-09
updated: 2026-10-09
tags: cloud proposal architecture-decision veo imagen gemini
sources:
  - mv_plan/google_cloud_integration.md
related:
  - sam3-microservice.md
  - ../pipelines/chained-generation.md
  - ../overview.md
---

# Decision: Optional Cloud Engines (PROPOSAL)

## Context

Local generation is the studio's religion — but some filmmakers want burst capacity
or models that don't run locally. The `mv_plan/google_cloud_integration.md` blueprint
proposes opt-in cloud engines that sit *alongside* the local stack, never in place
of it. Status: **plan/proposal, not implemented** (phased rollout described as
future work).

## Decision

Add per-engine cloud fallbacks, user-selected via a `CloudSettings.tsx` panel
(video engine, image engine, text engine dropdowns + secure credential fields),
with a backend "logic fork" so every router (`jobs.py`, `storyboard.py`,
`elements.py`) can dispatch to cloud worker modules (`tasks/video_cloud.py`,
`tasks/image_cloud.py`) instead of the local pipelines:

| Capability | Proposed cloud engine |
|---|---|
| Video generation | Google Veo 3.1 (Vertex AI), or "Nanobana" serverless GPU |
| Image gen / inpaint / upscale | Nano Banana Pro, Imagen 4 Ultra, `imagen-4.0-upscale-preview` |
| Script parsing + prompt enhancement | Gemini via Vertex AI |
| Semantic element matching | Vertex AI Text Embeddings API |

The hard rule: cloud tasks **completely bypass `job_utils.gpu_job_wrapper`** — the
local GPU hot path, semaphore, and memory slots are untouched. Local (LTX-2 / Flux 2 /
Ollama-Gemma) remains the default; cloud runs concurrently with local work and the
studio keeps working fully offline with zero cloud calls.

## Alternatives Considered

- **Local-only (status quo)**: kept as the default. The proposal doesn't replace it.
- **Cloud as a dependency**: rejected implicitly — the design goes out of its way to
  keep cloud optional per engine, which is the only form compatible with the
  local-first law.

## Consequences

- **Good**: burst capacity without new hardware; access to frontier models (Veo,
  Imagen) from inside the same timeline; local-first guarantee preserved by
  architecture, not just policy.
- **Cost**: credential management surface; two code paths per generation feature to
  maintain; cloud billing UX; the rollout is phased (backend infra first) and none
  of it is built yet.
- If implemented, the chained-generation "Cloud Hook" (§2.6 of the plan) would let
  long-video chunks render on cloud GPUs.

## Related

- [[sam3-microservice.md]] — the other "separate service" decision
- [[../pipelines/chained-generation.md]] — the cloud hook target
- [[../overview.md]] — the local-first premise this proposal must not break
