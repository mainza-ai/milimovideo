---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: sse events realtime jobs progress
sources:
  - docs/01_system_architecture.md
  - docs/05_execution_flow.md
related:
  - architecture.md
  - frontend-state.md
  - ../pipelines/inpainting.md
---

# SSE Eventing

## Purpose

The real-time nervous system: backend generation progress flows to the UI over
Server-Sent Events, with reconnect discipline and a polling fallback so refreshes
never lose track of running jobs.

## Components

| Component | Role |
|---|---|
| `backend/events.py` / `EventManager` | Per-client queue registry; workers push via `broadcast_progress()` / `update_job_db()` in `job_utils.py` |
| `GET /events` | The SSE endpoint (served by `server.py`, via `sse-starlette`), registered at startup |
| `SSEProvider.tsx` | React context owning the single `EventSource` lifecycle; exponential-backoff reconnect (1s → 10s max); replaced an older store-coupled inline listener |
| `hooks/useEventSource.ts` | Low-level hook returning `{lastEvent, isConnected}` |
| `utils/jobPoller.ts` | On page mount, finds every shot with `isGenerating: true` and re-syncs via `GET /status/{job_id}` — recovery after SSE dropouts or refresh |
| `ServerSlice.handleServerEvent` | Routes events into shot state (see [[frontend-state.md]]) |

## Key Interactions

**Event types and payloads:**

| Event | Payload | Meaning |
|---|---|---|
| `progress` | `{job_id, progress, statusMessage?, etaSeconds?, enhanced_prompt?}` | Denoising progress, e.g. "Loading models…", per-step %, per-chunk `chunk i/N` |
| `complete` | `{job_id, video_url, thumbnail_url}` (video); `{url, asset_id, type:"image"}` (image); `{job_id, thumbnail_url, type:"inpaint", asset_id}` (inpaint); `{shot_id, thumbnail_url, is_thumbnail:true}` (concept art) | Generation finished — matched by `lastJobId`, except thumbnails which match by `shot_id` |
| `error` | job id + message | Generation failed → toast |
| `cancelled` | `{job_id}` | Worker aborted → shot resets to pending, job lock cleared |
| `log` | (documented in 01; no payload examples in 05) | Backend log lines |

**Lifecycle:** worker emits → `EventManager` fans out to subscribed queues →
`SSEProvider` receives → `ServerSlice.handleServerEvent` updates the shot →
`triggerAssetRefresh()` bumps `assetRefreshVersion` so the MediaLibrary reloads.

**Recovery:** backend lifespan marks zombie `processing` jobs `failed` at startup
(never trust a job that outlived its process); frontend re-polls in-flight jobs on
mount; server shutdown calls `cancel_all_jobs()` plus a 2s grace so GPU threads can
escape.

## Gotchas

- There are no optimistic updates — if SSE drops and the poller misses, the UI can
  show a stale `isGenerating` until the next poll. The backoff + poller pairing is
  what makes this rare, not impossible.
- Cancellation is cooperative: `POST /jobs/{job_id}/cancel` sets
  `active_jobs[job_id]["cancelled"]`; the worker only notices at the next
  denoising-step callback, then raises `RuntimeError("Cancelled")`.
- Inpainting completions carry `type: "inpaint"` and an `asset_id`; the store must
  not confuse them with video completes.

## Audit Notes (2026-11-16)

See [[engineering-audit-2026-11.md]] for evidence:

- **`cancelled` is not reliably emitted for video.** `tasks/video.py` references an
  undefined `loop` inside the cancel branch and swallows the `NameError`
  (`video.py:785-790`). Image jobs do broadcast it; video jobs usually do not.
- **Chained cancellation can report success.** A pre-chunk cancellation `break`s
  out of the loop, then stitches the partial chunk list and marks the job
  `completed` (`chained.py:252`).
- **No replay.** A completion during an SSE outage is lost; `SSEProvider` reconnects
  but does not re-poll active jobs. `jobPoller` also has no `cancelled` branch.
- **Heartbeat timeout unused.** `lastEventTime` is stored but never enforced.
- **Inpaint `complete` has no `url`**, so `ServerSlice` cannot attach the result to
  the originating shot; it only appears in the gallery.
- The event table above says `video_url` for `complete`; the actual video payload
  uses `url`. Verify field names per producer.

## Related

- [[engineering-audit-2026-11.md]] — job/SSE findings
- [[architecture.md]] — the transport layer in context
- [[frontend-state.md]] — `ServerSlice` event handling
- [[../pipelines/inpainting.md]] — inpaint SSE payloads
