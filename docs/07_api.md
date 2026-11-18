# Milimo Video — Local API and Automation

Milimo Video is a local FastAPI service. Everything the Studio does is available
over HTTP, so batch generation and external tools use the same queue, memory
budget, and event stream as the UI.

- Base URL: `http://127.0.0.1:8000`
- Interactive schema: `http://127.0.0.1:8000/docs`
- OpenAPI document: `http://127.0.0.1:8000/openapi.json`

The work is serialized behind one GPU semaphore: jobs queue durably (one heavy
model at a time), and admission refuses a job before a load that cannot fit. Batch
scripts can submit many jobs; they run in priority/FIFO order.

## Authentication

The server binds to loopback by default and needs no token. If you set
`MILIMO_API_TOKEN` (required for a non-loopback bind), send a bearer token:

```bash
curl -H "Authorization: Bearer $MILIMO_API_TOKEN" \
  http://127.0.0.1:8000/health
```

`EventSource` cannot set headers, so the SSE endpoint also accepts
`?token=...`.

CORS is an allowlist (`MILIMO_CORS_ORIGINS`); add the origin of a browser client.
Uploads are capped by `MILIMO_MAX_UPLOAD_MB`.

## 1. Create a project

```bash
curl -s -X POST http://127.0.0.1:8000/projects \
  -H 'Content-Type: application/json' \
  -d '{"name":"API Batch","resolution_w":768,"resolution_h":512,"fps":25,"seed":42}'
```

The response includes `id`. A project is enough for ad-hoc jobs; a `Shot` row is
optional unless you want takes, review, and the timeline.

## 2. Queue a video job

`POST /generate/advanced` accepts an ad-hoc shot config. Omit `timeline` for
text-to-video, or add image conditioning items.

```bash
curl -s -X POST http://127.0.0.1:8000/generate/advanced \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "PROJECT_ID",
    "shot_config": {
      "id": "api-shot-1",
      "prompt": "A slow dolly-in on a rain-soaked street at night",
      "negative_prompt": "blurry, text, watermark",
      "seed": 42,
      "width": 768,
      "height": 512,
      "num_frames": 121,
      "num_inference_steps": 40,
      "enhance_prompt": false,
      "upscale": false,
      "pipeline_override": "auto"
    }
  }'
```

Response: `{"job_id": "job_...", "status": "queued"}`. Re-submitting the same
`shot_config.id` while it is pending/processing returns the existing job instead
of creating a second one.

## 3. Queue an image job

```bash
curl -s -X POST http://127.0.0.1:8000/generate/image \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "PROJECT_ID",
    "prompt": "A brass key on black velvet, macro photograph",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 8,
    "enable_ae": true
  }'
```

## 4. Preflight without loading a model

Check the measured budget and whether a preset fits before submitting a batch:

```bash
curl -s -X POST http://127.0.0.1:8000/system/memory/preflight \
  -H 'Content-Type: application/json' \
  -d '{"kind":"video","params":{"num_frames":121,"width":768,"height":512,"upscale":false}}'
```

The response has `ok`, `reason`, `retryable`, `estimate`, `downgrade`, and the
measured `snapshot`. A server-side refusal is still authoritative.

## 5. Follow progress

Polling:

```bash
curl -s http://127.0.0.1:8000/status/$JOB_ID
```

`status` is `queued|processing|completed|failed|cancelled`; `stage` is the current
stage and `memory` is the sampler trace.

Streaming (SSE):

```bash
curl -N http://127.0.0.1:8000/events
```

Events are typed `job.*` envelopes with `seq`, `stage`, `progress`, `step`,
`total`, `eta_seconds`, and exactly one terminal event. Reconnect with
`Last-Event-ID` (or `?last_event_id=`) to replay missed events.

## 6. Batch example (Python)

```python
import time
import httpx

BASE = "http://127.0.0.1:8000"
PROJECT = "PROJECT_ID"  # from POST /projects

shots = [
    {"id": "shot-1", "prompt": "A lighthouse in a storm", "seed": 1},
    {"id": "shot-2", "prompt": "A desert diner at dusk", "seed": 2},
]

with httpx.Client(base_url=BASE, timeout=30) as client:
    job_ids = []
    for shot in shots:
        response = client.post("/generate/advanced", json={
            "project_id": PROJECT,
            "shot_config": {
                "id": shot["id"],
                "prompt": shot["prompt"],
                "seed": shot["seed"],
                "num_frames": 49,
                "width": 512,
                "height": 320,
                "num_inference_steps": 8,
                "enhance_prompt": False,
                "upscale": False,
            },
        })
        response.raise_for_status()
        job_ids.append(response.json()["job_id"])

    for job_id in job_ids:
        while True:
            status = client.get(f"/status/{job_id}").json()
            print(job_id, status["status"], status.get("stage"), status.get("progress"))
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)
```

For a large batch, submit all jobs first and then poll: the durable queue keeps
the order and one GPU job runs at a time. A transient memory refusal appears as
`waiting` and is retried automatically; a permanent one fails with a reason.
Use `/system/memory/preflight` or a smaller preset for cards near the limit.

## 7. Cancellation and queue control

```bash
curl -X POST http://127.0.0.1:8000/jobs/$JOB_ID/cancel
curl -X POST http://127.0.0.1:8000/jobs/queue/pause
curl -X POST http://127.0.0.1:8000/jobs/queue/resume
curl http://127.0.0.1:8000/projects/$PROJECT_ID/active_jobs
```

A queued cancel clears both the `Job` and its dispatch row and emits one
`job.cancelled`. Pausing stops new dispatch after the current job.

## 8. Optional ComfyUI bridge

ComfyUI is an opt-in external engine. Core generation never calls it. With
`MILIMO_COMFY_ENABLED=0`, `/comfy/health` returns `enabled:false` without a network
probe, and `/comfy/run` / `/comfy/free` return 409.

```bash
curl -s http://127.0.0.1:8000/comfy/health
curl -s -X POST http://127.0.0.1:8000/comfy/validate \
  -H 'Content-Type: application/json' \
  -d '{"workflow":{"1":{"class_type":"EmptyImage","inputs":{"width":64,"height":48}},"2":{"class_type":"SaveImage","inputs":{"images":["1",0]}}}}'
```

`POST /comfy/run` takes `project_id`, optional `shot_id`, an API-format `workflow`,
optional per-node `overrides`, and optional `estimate_gb`/`estimate_seconds`. It
returns a normal `job_id`; use `/status/{job_id}` and the `job.*` stream. Outputs
are copied into the project and attached to the shot.

Set `MILIMO_COMFY_URL` if ComfyUI is not on `http://127.0.0.1:8001`, and
`MILIMO_COMFY_GB` to declare its residency. Settings are persisted locally and can
be read/changed over HTTP:

```bash
curl -s http://127.0.0.1:8000/comfy/settings
curl -s -X PUT http://127.0.0.1:8000/comfy/settings \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://127.0.0.1:8001","enabled":true,"declared_gb":2,"expected_revision":0}'
```

The read API redacts the bearer token (`has_token`); send `token` on PUT to
replace it and an empty string to clear it.

### ComfyUI discovery and media

```bash
curl -s http://127.0.0.1:8000/comfy/catalog
curl -s 'http://127.0.0.1:8000/comfy/schema?classes=CheckpointLoaderSimple,KSampler'
curl -s 'http://127.0.0.1:8000/comfy/models?folder=checkpoints'
curl -s -X POST http://127.0.0.1:8000/comfy/upload \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"PROJECT_ID","url":"/projects/PROJECT_ID/assets/general/source.png"}'
```

`validate` accepts `live:true` for `/object_info` preflight:

```bash
curl -s -X POST http://127.0.0.1:8000/comfy/validate \
  -H 'Content-Type: application/json' \
  -d '{"live":true,"workflow":{"1":{"class_type":"EmptyImage","inputs":{"width":64,"height":48}},"2":{"class_type":"SaveImage","inputs":{"images":["1",0],"filename_prefix":"x"}}}}'
```

### Saved workflows

```bash
curl -s http://127.0.0.1:8000/projects/PROJECT_ID/comfy/workflows
curl -s -X POST http://127.0.0.1:8000/projects/PROJECT_ID/comfy/workflows \
  -H 'Content-Type: application/json' \
  -d '{"name":"Smoke","workflow":{...},"bindings":{"width":{"node":"1","input":"width"}}}'
```

Update with `expected_revision`; a stale write returns 409. `POST
/projects/{id}/comfy/workflows/{workflow_id}/default` sets the project default;
`DELETE` removes it. `POST /comfy/run` accepts `workflow_id`, `workflow_name`,
`bindings`, `estimate_seconds`, and `shot_id`.

## 9. Render, titles, and grading

`POST /projects/{project_id}/render` accepts `settings.lut` (a project `.cube`/
`.3dl`/`.dat`/`.m3d` URL), `start_frame`/`end_frame`, `fit`, and encoding fields.
Project `titles` are a list of `{text,start_frame,end_frame,x,y,size,color,box}` and
are burned in after the LUT, before captions. Shot `transition_in` is a V1
crossfade duration in frames. Set `pipeline_override: "ltx25"` for the gated
LTX-2.5 diffusers engine (default path `backend/models/ltx-2.5-diffusers`). `GET /projects/{id}/otio?format=otio|fcpxml` exports
the V1/V2/A1/A2 cut; `POST /projects/{id}/otio/import` accepts OTIO or FCPXML.

## 10. Delete assets

Library cards expose **Delete**, which calls `DELETE /assets/{asset_id}`. The
endpoint removes the asset file and poster, clears shot `video_url`/`thumbnail_url`
references, and marks matching jobs deleted. `DELETE /upload/{filename}` remains
for legacy filename-based callers.

## 11. Automation guardrails

- The API is local-first. Cloud providers remain off unless `MILIMO_LLM_CLOUD=1`
  and air-gap is off; a batch cannot silently spend money.
- Generated media is written under `projects/{project_id}/generated/` and served
  through `/projects/...`.
- Long batches should poll `/metrics/jobs` for queue wait, memory counters, and
  recent failures.
- A terminal webhook can be configured with `MILIMO_JOB_WEBHOOK` (and
  `MILIMO_JOB_WEBHOOK_TOKEN`) when a filesystem watcher is not enough.

## Related

- `docs/01_system_architecture.md` — routers and execution flow
- `docs/comfyui.md` — optional ComfyUI bridge setup and troubleshooting
- `Milimo Video/systems/sse-eventing.md` — event envelope and replay
- `Milimo Video/systems/memory-management-audit-2026-11.md` — budget/admission
