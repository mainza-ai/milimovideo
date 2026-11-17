---
type: systems
created: 2026-10-09
updated: 2026-11-17
tags: sse events realtime jobs progress queue
sources:
  - ../backend/events.py
  - ../backend/job_utils.py
  - ../web-app/src/providers/SSEProvider.tsx
  - ../web-app/src/studio/
related:
  - architecture.md
  - frontend-state.md
  - ../pipelines/inpainting.md
---

# SSE Eventing

## Purpose

The real-time nervous system: backend generation, render, depth, model-download, and
queue events flow to the UI over Server-Sent Events, with polling recovery so a
refresh or dropped stream does not leave a permanent spinner.

> **Status (2026-11-17):** the transport works, but the Studio shell does not
> consume it and has no toast layer, so generation progress is invisible there.
> The audit and plan are in [[job-notifications-audit-2026-11]].

## Components

| Component | Role |
|---|---|
| `backend/events.py` / `EventManager` | Per-client queue registry; one broadcast fans out to subscribers |
| `GET /events` | SSE endpoint served by `server.py` via `sse-starlette` with heartbeat |
| `SSEProvider.tsx` | Global EventSource lifecycle, but dispatches **only** into the legacy timeline store; Studio has no consumer (N-02 in [[job-notifications-audit-2026-11]]) |
| `jobPoller.ts` | Re-syncs shots still flagged `isGenerating` through `GET /status/{job_id}` |
| Studio hooks | Poll `/status/{job_id}` for preview/depth/inpaint and `/active_jobs` every 2.5 s (Generate only); renders every 4 s. The main generation flow has no fallback (N-03/N-08) |
| `job_utils.py` | Queue ledger, GPU semaphore, OOM retry, pause/resume, progress/log broadcasts |

## Event types

| Event | Payload | Meaning |
|---|---|---|
| `progress` | `{job_id, progress, status, status_message?, eta_seconds?, enhanced_prompt?}` | Denoising/queue progress |
| `complete` | `{job_id, url, thumbnail_url?, type?, asset_id?}` | Generation finished |
| `error` | `{job_id, message}` | Generation failed |
| `cancelled` | `{job_id}` | Worker aborted |
| `log` | `{job_id, message}` | Backend log line |
| `render_progress` | `{job_id, project_id, progress, message, status}` | Render progress |
| `render_completed` / `render_failed` / `render_cancelled` | Render payload with `video_url`/`error` | Render terminal state |
| `depth_complete` / `depth_failed` | `{job_id, project_id, url}` / error | Depth overlay |
| `ai_parse_complete` / `ai_parse_failed` | parse result/error | Queued AI script parse |
| `download_queued`, `download_progress`, `download_complete`, `download_error`, `queue_updated` | model download state | Model downloads |
| `model_status_change`, `lora_updated` | model/LoRA state | Setup and activation |

## Lifecycle

1. A request creates a `Job` and enqueues a `QueuedJob` — **no event** is emitted
   for admission or position (N-04).
2. `QueuedJob` waits for the GPU semaphore (and for pause to clear) — **no event**.
3. The task registers `active_jobs` and broadcasts `progress`/`log`; terminal state
   is written and broadcast by the task, but not by every path (N-12).
4. Only the legacy store consumes the stream; Studio polls three specific flows and
   ignores the rest. Toasts exist only in the legacy shell (N-01).

## Recovery and cancellation

- Startup recovery marks stuck `running` ledger rows retryable once, fails ones that
  already retried twice, and resumes `queued` rows.
- `POST /jobs/{job_id}/cancel` sets the cancellation flag for active work, or marks
  a pending job cancelled in the DB. The pending path emits **no event**, leaves the
  `QueuedJob` row queued, and does not set `completed_at` (N-05/N-16).
- Render cancellation terminates FFmpeg and emits exactly one terminal
  `render_cancelled` event.
- The legacy SSE provider re-polls in-flight shots on reconnect. Studio has no
  reconnect resync; a reload during generation shows the shot as idle (N-08). There
  is no `Last-Event-ID` replay (N-24).
- Queue pause stops new dispatch after the current job; resume releases waiting
  jobs within ~250ms. Health reports `queue_paused`.

## Gotchas

- **Listener coverage is incomplete even in the legacy app.** `SSEProvider` never
  registers `cancelled`, `ai_parse_*`, `depth_*`, `download_queued`,
  `queue_updated`, `lora_updated`, or `render_*`. `serverSlice` has a `cancelled`
  handler that can never fire (N-15).
- **Studio addToast calls are silent.** 104 call sites write to a store whose toast
  container is not mounted (N-01).
- Terminal state must be written once and broadcast once; duplicate generic
  `cancelled` plus typed terminal events were removed for render, but there is no
  single terminal emitter to enforce it across tasks (N-12).
- A `render_*` event is the render path; the generic `complete`/`error` events are
  generation paths.
- Model downloads and shots can emit events concurrently; match by `job_id` or the
  relevant model/shot ID, never by position in the stream.
- A 404 from `GET /status/{job_id}` means the job is not running; clear transient
  state instead of polling forever.
- SSE is local-only by default; if a bearer token is enabled, EventSource carries it
  via the query parameter because it cannot set headers.
- Heartbeats keep proxies from closing the stream; a missing heartbeat is a signal
  to reconnect/poll.

## Related

- [[job-notifications-audit-2026-11.md]] — audit, gap register, and N0–N5 plan
- [[architecture.md]] — transport layer in context
- [[frontend-state.md]] — how events reach Studio/legacy state
- [[../pipelines/inpainting.md]] — inpaint payloads
- [[../roadmap/implementation-plan-2026-11.md]] — cancellation/recovery status
