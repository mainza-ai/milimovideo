---
type: systems
created: 2026-10-09
updated: 2026-11-17
tags: sse events realtime jobs progress queue memory notifications
sources:
  - ../../backend/events.py
  - ../../backend/job_events.py
  - ../../backend/job_utils.py
  - ../../web-app/src/providers/SSEProvider.tsx
  - ../../web-app/src/studio/events/jobBridge.ts
  - ../../web-app/src/studio/components/StudioToastLayer.tsx
  - ../../web-app/src/studio/components/ActivityCenter.tsx
related:
  - architecture.md
  - frontend-state.md
  - job-notifications-audit-2026-11.md
  - ../pipelines/inpainting.md
---

# SSE Eventing

## Purpose

The real-time nervous system: generation, render, depth, model-download, queue, and
memory events flow to both the Studio and the frozen legacy editor over one
Server-Sent Events connection, with sequence ids and replay so a reconnect does not
lose a terminal state.

> **Status (2026-11-17):** implemented. Studio consumes the stream, renders inline
> progress/toasts/activity, and replays missed events. The audit and evidence live
> in [[job-notifications-audit-2026-11]].

## Components

| Component | Role |
|---|---|
| `backend/events.py` / `EventManager` | Per-client unbounded queues, monotonic `seq`, a 1000-message / 120 s replay ring, dropped/peak-pending stats |
| `backend/job_events.py` | Typed envelope, `register`, stage/progress helpers, one `terminal()` that writes the DB once and emits once, optional webhook |
| `GET /events` | SSE endpoint; accepts `Last-Event-ID` (header) or `last_event_id` (query) and replays newer messages |
| `SSEProvider.tsx` | One EventSource for the app, exponential backoff, `job.*` → Studio bridge + legacy mapping, system handlers, resync on connect |
| `jobBridge.ts` | Maps a job to its shot, updates stage/progress/ETA/take, emits toasts, desktop notifications, and a `milimo:job-terminal` event |
| `jobStore.ts` | Active jobs, history, unread count, metrics |
| `StudioToastLayer.tsx` | The single mounted toast renderer (aria-live, dismiss, tone) |
| `ActivityCenter.tsx` | TopBar active/history popover, cancel/retry, document title and dock badge |
| `jobPoller.ts` | Legacy recovery path; Studio reconciles through `/active_jobs` + `/status` on mount/resync |

## Event envelope

Every job event carries:

```json
{
  "ts": 1794970000.0,
  "event": "job.stage",
  "job_id": "job_633f998f",
  "shot_id": "shot-1",
  "project_id": "33bd2bbe",
  "kind": "video",
  "stage": "denoising",
  "progress": 42,
  "step": 7,
  "total": 20,
  "eta_seconds": 31,
  "message": "Denoising 7/20"
}
```

Terminal payloads add `url`, `thumbnail_url`, `actual_frames`, `error`, `memory`,
or `asset_id` as applicable.

## Event types

| Event | Meaning |
|---|---|
| `job.queued` | Dispatch row created; includes position and estimate |
| `job.waiting` | Paused queue or transient memory refusal; includes reason |
| `job.started` | Worker picked the row up |
| `job.stage` | Named stage change (enhancing/loading/conditioning/denoising/upscaling/inpainting/encoding/saving) |
| `job.progress` | Progress without a stage change |
| `job.retrying` | OOM retry or memory downgrade applied |
| `job.cancelling` | Active job saw the cancel flag |
| `job.completed` / `job.failed` / `job.cancelled` | Terminal; emitted at most once per job |
| `render_*`, `depth_*`, `ai_parse_*`, `download_*`, `queue_*`, `model_status_change`, `lora_updated` | Non-job surfaces |
| `memory.pressure`, `memory.evicted` | Budget monitor and manual/automatic release |
| `progress` / `complete` / `error` / `cancelled` | Legacy-compatible mapping produced by `SSEProvider` for the frozen editor |

## Lifecycle

1. A request creates a `Job` and `QueuedJob`; `enqueue_gpu_job` emits `job.queued`
   with position and estimate.
2. `run_queued_job` respects the paused queue, then `wait_for_turn` orders dispatch
   by priority, then by the resident slot, then FIFO. The row is marked running and
   `job.started` is emitted.
3. Tasks emit `job.stage`/`job.progress` with step/total/ETA; the sampler records a
   per-stage peak.
4. A terminal path calls `job_events.terminal(...)`, which releases the lease, stops
   the sampler, writes the `Job`/`memory_json`, and emits exactly one terminal event.
5. Studio's `jobBridge` updates the shot, job store, toast layer, and activity
   center; `SSEProvider` also maps the event into the legacy store.

## Recovery and cancellation

- Reconnect passes the last seen id; the server replays messages newer than it from
  the 120 s ring. A reload also triggers `milimo:resync`, which refreshes
  `/active_jobs` and reconciles shots against `/status`.
- Startup recovery fails rows whose worker died (`stage=recovery`) and emits a
  terminal event instead of re-running a heavy job. The OOM retry re-admits the
  reduced preset first.
- `POST /jobs/{job_id}/cancel`: active work gets `job.cancelling` and converges via
  the task callback; a queued job cancels both `Job` and `QueuedJob`, sets
  `completed_at`, and emits `job.cancelled` immediately.
- A transient memory refusal emits `job.waiting`, stores a retry time on the queue
  row, and re-dispatches with backoff (20 attempts) without holding the semaphore.
- Render cancellation terminates FFmpeg and emits exactly one `render_cancelled`.

## Gotchas

- The terminal emitter is the only writer of terminal state; adding a second
  `event_manager.broadcast("complete")` would create duplicate UI updates.
- `EventManager` queues are unbounded on purpose: dropping a terminal event is worse
  than a little memory. `event_manager.stats()` exposes `dropped`/`peak_pending`.
- The native EventSource `error` event has no data; handlers must ignore empty
  payloads and let `onerror` own reconnection.
- A 404 from `GET /status/{job_id}` means the job is unknown; clear the spinner.
- Match by `job_id`/`shot_id`/model id, never by stream position.
- If a bearer token is enabled, EventSource carries it via the query parameter
  because it cannot set headers.
- Heartbeats keep proxies from closing the stream; a missing heartbeat is a signal
  to reconnect.
- The legacy editor only receives the mapped names; it does not render Studio
  toasts or the activity center.

## Related

- [[job-notifications-audit-2026-11.md]] — audit, gap register, N0–N5 status, live probes
- [[architecture.md]] — transport layer in context
- [[frontend-state.md]] — how events reach Studio/legacy state
- [[../pipelines/inpainting.md]] — inpaint payloads
- [[../roadmap/implementation-plan-2026-11.md]] — cancellation/recovery status
