---
type: decisions
created: 2026-10-09
updated: 2026-10-09
tags: sam-3 microservice architecture-decision
sources:
  - docs/01_system_architecture.md
  - docs/03_ai_pipelines.md
related:
  - ../systems/architecture.md
  - ../models/sam-3.md
---

# Decision: SAM 3 as an Isolated Microservice

## Context

SAM 3 (segmentation + video tracking) has dependency and device needs that clash
with the main backend: its own patched predictor code (device-param fixes,
MPS workarounds), a separate conda env (`sam3_env`, Python 3.12), and a ~3.4GB
checkpoint with lazy-loaded video predictor state. The main backend already juggles
LTX-2 and Flux 2 under strict memory-slot discipline.

## Decision

Run SAM 3 as its own FastAPI service on port 8002 (`sam3/start_sam_server.py`),
started via `./run_sam.sh`. The backend talks to it over HTTP (`InpaintingManager`,
`TrackingManager` as clients). Segmentation/tracking is optional — the studio
generates video fine with the SAM service down.

## Alternatives Considered

- **In-process import**: would force SAM 3's patched torch code and env into the
  backend process, risking dependency conflicts with the LTX-2/Flux 2 stack and
  complicating the memory-slot accounting (a third giant in one process).
- **Cloud segmentation API**: rejected outright — violates the local-first religion.
  A filmmaker's masks shouldn't require someone else's server.

## Consequences

- **Good**: clean failure boundary (segmentation down ≠ studio down); independent
  device handling (CUDA → MPS → CPU auto-detect); separate env avoids dependency
  hell; readiness probe via `GET /health`.
- **Cost**: one more service to launch; HTTP + multipart overhead per mask call;
  tracking sessions are stateful on the SAM side (`session_id`) and must be
  explicitly closed.

## Related

- [[../systems/architecture.md]]
- [[../models/sam-3.md]] — the API surface
