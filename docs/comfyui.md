# Using the ComfyUI Bridge

ComfyUI is an **optional** local engine. Milimo Video does not ship, start, or
require it. Core generation (LTX-2, Wan, Flux, SAM, render, OTIO) works without it.
The bridge exists so an API-format ComfyUI workflow can run through Milimo's queue,
show the same progress UI, and land its outputs in the current project.

## 1. Start ComfyUI

Run ComfyUI however you normally do (Comfy Desktop or `python main.py`). The
default Milimo URL is:

```text
http://127.0.0.1:8001
```

If ComfyUI is elsewhere, set the URL before starting the backend:

```bash
export MILIMO_COMFY_URL=http://127.0.0.1:8188
```

SAM 3 defaults to port **8002**, so ComfyUI can keep 8001. To hide the bridge
entirely:

```bash
export MILIMO_COMFY_ENABLED=0
```

`MILIMO_COMFY_GB` declares ComfyUI's residency for admission (default 2 GB); set
it to the peak of the workflows you run.

## 2. Export an API-format workflow

In ComfyUI, build the graph, then choose:

```text
Workflow -> Save (API Format)
```

Milimo needs the API JSON (`{"1": {"class_type": ..., "inputs": ...}}`), not the
UI JSON (`nodes` / `links`). The workflow must contain at least one `SaveImage`
or other `Save*`/`Preview*` node, otherwise it produces no files and Milimo
rejects it.

## 3. Run it from Studio

1. Open **Generate** and select the target shot.
2. Expand **ComfyUI workflow**.
3. Paste the API JSON or click **Import JSON**.
4. Click **Validate**. It reports node count, output nodes, and inferred
   resolution without contacting ComfyUI.
5. Click **Run workflow**.

The job enters the normal durable GPU queue. Progress appears as queue position,
node execution, sampler steps, and saving; **Cancel** deletes/interrupts the
ComfyUI prompt. Completed outputs are copied into
`projects/<project>/generated/`, inserted into Library as assets, and the primary
image/video is attached to the shot. Video outputs get a poster frame.

The workflow draft is kept in browser `localStorage` (`milimo-comfy-workflow`).

## 4. Smoke test without models

ComfyUI can validate a bridge with no checkpoints:

```json
{
  "1": {
    "class_type": "EmptyImage",
    "inputs": {"width": 640, "height": 360, "batch_size": 1, "color": 0}
  },
  "2": {
    "class_type": "SaveImage",
    "inputs": {"images": ["1", 0], "filename_prefix": "milimo"}
  }
}
```

Click **Smoke test** in the panel, then **Run workflow**. A PNG should appear in
the project.

## 5. API

```bash
# Probe
curl -s http://127.0.0.1:8000/comfy/health

# Validate without a network call
curl -s -X POST http://127.0.0.1:8000/comfy/validate \
  -H 'Content-Type: application/json' \
  -d '{"workflow": {"1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 48}}, "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}}'

# Queue
curl -s -X POST http://127.0.0.1:8000/comfy/run \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "PROJECT_ID",
    "shot_id": "SHOT_ID",
    "workflow": {"1": {"class_type": "EmptyImage", "inputs": {"width": 640, "height": 360}}, "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}},
    "estimate_seconds": 30
  }'

# Follow the ordinary job stream / poll
curl -s http://127.0.0.1:8000/status/JOB_ID
curl -N http://127.0.0.1:8000/events
```

`POST /comfy/run` also accepts `overrides` (`{"NODE_ID": {"input": value}}`) to
reuse a saved graph with a new prompt/seed, and `filename_prefix`. `POST
/comfy/free` asks ComfyUI to release cached models; it refuses while a workflow is
executing.

## 6. Memory and scheduling

ComfyUI is an external GPU process. Milimo:

- admits the declared `MILIMO_COMFY_GB` before submitting;
- queues the workflow as kind `comfy` alongside video/image jobs;
- asks ComfyUI `/free` to release its cache when another engine needs the slot;
- never frees ComfyUI while its own `/queue` reports a running prompt.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ComfyUI is not reachable` | Start ComfyUI or set `MILIMO_COMFY_URL`; check `MILIMO_COMFY_ENABLED` |
| `This looks like a UI workflow` | Re-export with **Save (API Format)** |
| `no Save/Preview node` | Add `SaveImage` (or another Save node) |
| Panel missing in Generate | Bridge disabled, or ComfyUI offline; check **Setup -> ComfyUI bridge** |
| `node_errors` | Open the same API JSON in ComfyUI or check missing models/custom nodes |
| No result files | The graph's Save node did not execute, or wrote nothing |
| Port conflict | ComfyUI uses 8001, SAM 3 uses 8002; override either with env vars |

## Related

- [User Guide](user_guide.md) — Studio-first walkthrough
- [Local API](07_api.md) — job, SSE, and ComfyUI endpoints
- `Milimo Video/systems/comfyui-bridge.md` — implementation and internal contract
