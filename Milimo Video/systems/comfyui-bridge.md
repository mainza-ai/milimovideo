# ComfyUI bridge

**Status:** implemented and live-verified against ComfyUI 0.45.0 on
`http://127.0.0.1:8001` with a model-free `EmptyImage → SaveImage` graph.

## What it is

A generic bridge from Milimo Video to a local ComfyUI server. Milimo does not
try to mirror ComfyUI's node graph editor. It accepts a **ComfyUI API-format
workflow** (the JSON written by *Workflow → Save (API Format)*), submits it,
relays progress, and ingests every output file into the project workspace.

No checkpoint, LoRA, sampler, or model name is hardcoded. The local ComfyUI
instance owns its model library; Milimo only owns the job lifecycle and the
project copy of the result.

## Contract

| Surface | Purpose |
|---|---|
| `GET /comfy/health` | Probe `/system_stats` and `/queue`; never fails when ComfyUI is down (`reachable: false`) |
| `POST /comfy/validate` | Structural checks without a network call: node count, output nodes, inferred width/height |
| `POST /comfy/run` | Validate, create a `Job(type="comfy")`, enqueue a durable GPU job |
| `POST /comfy/free` | Call ComfyUI `/free` and release the `comfy` residency slot |

`POST /comfy/run` body:

```json
{
  "project_id": "...",
  "shot_id": "optional-shot",
  "workflow": {"1": {"class_type": "...", "inputs": {}}},
  "overrides": {"3": {"text": "a prompt"}},
  "filename_prefix": "optional",
  "estimate_gb": 2.0,
  "estimate_seconds": 30
}
```

`workflow` may be a JSON object or a JSON string. `overrides` merges per-node
`inputs` after parsing, which is how the same saved graph is reused with a new
prompt or seed. UI-format workflows (`{"nodes": [...], "links": [...]}`) are
rejected with an actionable message.

## Execution path

1. **Admission.** `memory_manager.admit_job(job_id, "comfy", params)` budgets
   `MILIMO_COMFY_GB` (default 2 GB, +1 GB workspace, ×1.05 safety). A refusal is
   a `MemoryDeferred`, so the durable queue waits and retries rather than failing
   the user's job. The `comfy` slot leases the external process for the run.
2. **Submit.** `ComfyClient.submit` posts `{"prompt": workflow, "client_id": uuid}`
   to `/prompt`. Empty `node_errors` and a `prompt_id` are required.
3. **Progress.** A websocket to `/ws?clientId=...` maps `execution_start`,
   `executing`, `progress`, `progress_state` (ComfyUI 0.4x), `executed`,
   `execution_success`, `execution_error`, and `execution_interrupted` to typed
   `job.*` stages. If `websockets` is missing or the socket drops, a
   queue/history heartbeat keeps progress moving, so completion never depends on
   the socket.
4. **Outputs.** `/history/{prompt_id}` is polled until the entry appears. Every
   `outputs[*]` list item with a `filename` is streamed from
   `/view?filename=&subfolder=&type=` directly to
   `projects/<project>/generated/comfy_<job>_<n>_<stem><ext>`.
5. **Ingestion.** Each image/video/audio/text output becomes an `Asset` row
   (Library) with engine/source metadata; video outputs get a poster frame. The
   primary image/video is written to the shot (`video_url`, `thumbnail_url`,
   `last_job_id`, `status`); video outputs also become the selected take.
6. **Terminal.** One `job.completed/failed/cancelled` event carries the output
   URL, thumbnail, assets, ComfyUI `prompt_id`, validation readout, and elapsed
   seconds. `terminal` releases the memory lease.

Cancellation checks `active_jobs[job_id]["cancelled"]` every poll, deletes the
queued prompt, and interrupts it when running (`/queue` + `/interrupt`).

## Memory and port rules

ComfyUI is a separate GPU process with no per-model accounting. It is treated
as a declared external engine:

- `MILIMO_COMFY_GB` is the declared residency (default 2 GB).
- The `comfy` slot conflicts with `video`, `image`, and `llm_http`; those loads
  ask ComfyUI to `/free` its cache when the slot is idle, and a ComfyUI run
  unloads conflicting Milimo slots through the normal admission path.
- ComfyUI is included in the single global GPU queue as `kind="comfy"`
  (priority 20, same tier as video/image).

**Port collision:** Comfy Desktop defaults to `8001`, which is also the default
for the optional SAM 3 microservice. Set `MILIMO_COMFY_URL` (for example
`http://127.0.0.1:8188`) when both must run. SAM is optional and lazy-loaded.

## UI

- **Setup → ComfyUI bridge:** reachability, version, queue depth, free RAM,
  probe, free cached models, open ComfyUI.
- **Generate → ComfyUI workflow:** paste or import API-format JSON, one-click
  model-free smoke graph, validate, run against the selected shot, cancel. The
  last workflow is kept in `localStorage` (`milimo-comfy-workflow`).
  Per-step progress renders through the existing activity center and shot
  progress, not a separate poller.

## Verification

```bash
./milimov/bin/python -m pytest tests/test_comfy_bridge.py
```

Covers API/UI workflow parsing, overrides/prefix, validation, fake-server
submit/history/download, rejection and offline health, `progress_state` mapping,
route errors, plus a live end-to-end run (skipped when ComfyUI is offline) that
asserts a copied file, asset row, shot update, and terminal progress.

Observed live: ComfyUI 0.45.0, 2082 nodes, no checkpoints; the model-free
`EmptyImage → SaveImage` graph completed in about one second and produced a
64×48 PNG copied into the project.

## Limitations

- Node-graph editing is not reimplemented; ComfyUI remains the editor.
- ComfyUI residency is declared, not measured per model. Set
  `MILIMO_COMFY_GB` to match the largest workflow the user runs.
- A workflow can reference paths the user could also load manually in ComfyUI;
  the bridge adds no new filesystem surface.
- WebSocket progress can miss the first events of a sub-second graph; the
  polling heartbeat still reports completion.
- Outputs are copied, not linked. Deleting the project copy does not delete the
  ComfyUI output directory.
