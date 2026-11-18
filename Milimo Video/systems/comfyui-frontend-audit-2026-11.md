---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: comfyui frontend audit workflow integration roadmap
sources:
  - ../../web-app/src/studio/components/ComfyUIPanel.tsx
  - ../../backend/routes/comfy.py
  - ../../backend/services/comfy.py
  - ../../backend/tasks/comfy.py
  - ../../tests/test_comfy_bridge.py
related:
  - comfyui-bridge.md
  - ../roadmap/comfyui-frontend-plan-2026-11.md
  - ../roadmap/open-work-resolution-2026-11.md
---

# ComfyUI Frontend Integration Audit (2026-11-18)

## Verdict

The ComfyUI backend is a usable **generic workflow bridge**. The frontend is a
**minimal paste-JSON runner**, not yet a production ComfyUI integration. The main
gaps are not transport or output ingestion; they are workflow binding, discovery,
persistence, validation, project-media handoff, and result selection.

Current state: **backend ahead of frontend by roughly one product phase**.

## Method

- Reviewed `ComfyUIPanel.tsx`, `ComfyHealthPanel`, `routes/comfy.py`,
  `services/comfy.py`, `tasks/comfy.py`, and their tests.
- Probed the live local ComfyUI 0.45.0 with 2,082 object-info nodes and 11 workflow
  templates.
- Called `/comfy/health`, `/comfy/validate`, `/comfy/run`, and the job plane.
- Verified the disabled/offline paths and the model-free smoke graph.

## Current capability matrix

| Area | Current behavior | Verdict |
|---|---|---|
| Discovery | Health, version, queue depth, free RAM, Open ComfyUI | Basic |
| Workflow input | Paste/import one API-format JSON; smoke graph | Functional |
| Validation | Local shape checks (nodes, Save/Preview nodes, inferred size) | Structural only |
| Execution | Selected shot, durable queue, admission, cancel, terminal event | Solid |
| Progress | WebSocket steps/node names, polling fallback, global activity center | Good backend, thin UI |
| Outputs | Stream files into project, Asset rows, shot attachment, poster | Solid |
| Persistence | One browser localStorage key for the whole app | Weak |
| Setup | Env-only URL/GB/enabled; probe/free/open | Basic |
| Diagnostics | Raw backend errors; no node/model preflight | Weak |

Live proof that structural validation is too permissive:

```json
{"1":{"class_type":"DefinitelyMissingNode","inputs":{"x":1}},
 "2":{"class_type":"SaveImage","inputs":{"images":["1",0]}}}
```

`/comfy/validate` returns `{"ok":true,"nodes":2,"output_nodes":["2"]}`. The missing
node is only discovered when ComfyUI rejects `/prompt`.

## Gap register

### P1 — blocks normal use

| ID | Gap | Evidence | Impact |
|---|---|---|---|
| C-01 | No workflow parameter surface; one raw textarea | `ComfyUIPanel.tsx:307-317` | Users must hand-edit JSON to change a prompt, seed, size, or model |
| C-02 | No prompt/seed binding from the shot | `run()` posts only `project_id`, `shot_id`, `workflow` at `ComfyUIPanel.tsx:227-235`; API supports `overrides` at `routes/comfy.py:104-129` | Every shot needs a new JSON; no reusable graph |
| C-03 | No project-media → ComfyUI input bridge | No `/upload/image` call in `services/comfy.py` or routes; `LoadImage` workflows require files already in Comfy's `input/` | Reference/start-frame workflows are impractical |
| C-04 | Validation does not check node classes or required inputs | `/comfy/validate` only calls `parse_workflow` + `validate_workflow` (`routes/comfy.py:52-63`); live missing-node probe passed | Failures surface only after a queued job starts |
| C-05 | ComfyUI capabilities are not discovered | `ComfyClient.object_info()` exists (`services/comfy.py:284`) but is unused; `/workflow_templates` is never called | No node/model/template pickers; users guess class and checkpoint names |
| C-06 | Workflow state is global, not project/shot scoped | `WORKFLOW_STORAGE_KEY` is one key (`ComfyUIPanel.tsx:41,176-181,238-242`) | Switching projects/shots overwrites the only draft |
| C-07 | No output preview or primary-output selection | Task picks first video/image (`tasks/comfy.py:243-246`); panel has no result list | Multi-output graphs attach an arbitrary result; no quick review |
| C-08 | Reload loses the panel's job identity and status | `jobId` is local state (`ComfyUIPanel.tsx:185`); no recovery from active jobs | Cancel/progress controls disappear after refresh |

### P2 — product depth and trust

| ID | Gap | Evidence | Impact |
|---|---|---|---|
| C-09 | No workflow export, copy, format, or line numbers | Only Import/Smoke/Validate/Run (`ComfyUIPanel.tsx:318-369`) | Hard to iterate or share graphs |
| C-10 | No workflow library, versions, tags, or templates | Only localStorage; Comfy exposes 11 templates but none are consumed | Reuse depends on copy/paste from external files |
| C-11 | No node-level progress UI | Backend maps websocket node names/steps; panel shows no inline job state (`ComfyUIPanel.tsx:370-385`) | Progress only in the global activity center |
| C-12 | No error taxonomy for missing models/custom nodes | Raw `node_errors` text is returned/truncated | User cannot tell a missing checkpoint from a bad input |
| C-13 | Setup config is env-only | `ComfyHealthPanel` only probes/free/open (`ComfyUIPanel.tsx:91-169`) | Changing URL/GB/token requires restarting the backend |
| C-14 | No auth/TLS/remote-Comfy posture | `MILIMO_COMFY_URL` is used directly; no token handling | Remote or secured Comfy instances are unsupported |
| C-15 | Workflow provenance is incomplete on assets/takes | Asset meta stores engine/job/node/source/prompt (`tasks/comfy.py:119-125`); no workflow JSON, overrides, prompt_id, or version | Cannot reproduce a take from its asset |
| C-16 | No Comfy queue controls or manual-job coordination | Milimo cancels only its own job; `/queue`/`/interrupt` are internal | Manual Comfy jobs can surprise the Milimo queue |
| C-17 | No batch/workflow matrix | One run at a time through the panel | No prompt/seed sweep for a graph |
| C-18 | No input/output size or node-count limits in the UI | Validation has no limits; output download is unlimited | Accidental multi-GB graphs or outputs are easy |

### P3 — polish, accessibility, tests

| ID | Gap | Impact |
|---|---|---|
| C-19 | JSON textarea has no syntax highlighting, folding, or error line/column | Hard to edit large graphs |
| C-20 | No keyboard shortcut for validate/run; no focus management after errors | Accessibility/fast iteration |
| C-21 | No explicit “unsaved workflow” state or dirty indicator | Users lose edits on project switch |
| C-22 | Frontend tests cover only health/validate/run/disabled; no asset upload, overrides, persistence, output selection, or E2E | Regressions in the hard parts go unnoticed |
| C-23 | No ComfyUI version-compatibility gate | New/old Comfy APIs can fail late |
| C-24 | Panel hidden when offline; no “open Setup” affordance in Generate | Discovery is weak for first-time users |

## What is already good

- Optional-by-design: `MILIMO_COMFY_ENABLED=0` removes the surface; core paths never
  import `services.comfy`.
- Durable execution through `QueuedJob`, memory admission, one global GPU slot,
  cancellation, `/free`, and one terminal event.
- Streamed output ingestion with sanitized filenames, Asset rows, poster frames,
  and shot attachment.
- WebSocket progress with polling fallback and `progress_state` support.
- A real live smoke test and fake-server unit coverage.
- Clear docs for setup, smoke graph, API, and troubleshooting.

## Recommended enhancements

1. **Make a workflow reusable, not a JSON blob.** Introduce a binding object:
   `{workflow, bindings: {prompt: "6.text", seed: "3.seed", width: "5.width",
   model: "4.ckpt_name"}, defaults}`. Draw the controls from `object_info`, not
   hardcoded field names.
2. **Bridge project media.** Add `POST /comfy/upload` that proxies an asset to
   ComfyUI `/upload/image`, returns the Comfy input name, and rewrites bound
   `LoadImage` nodes. Clean up temporary inputs after the job.
3. **Preflight against the live graph.** Use `/object_info` to verify every
   `class_type`, required input, enum choice, and model filename. Surface
   `node_errors` as per-node rows before queueing.
4. **Persist workflows.** Store named workflows per project (and optional shot
   binding) with revision, graph JSON, binding map, and a “set default” flag.
   Keep localStorage only as an unsaved draft.
5. **Treat outputs as a result set.** Show thumbnails, type, node, dimensions, and
   “Use as take” for each output; default the primary to the highest-priority
   Save node, not array order.
6. **Put provenance on the take.** Record workflow hash/name/version, overrides,
   Comfy `prompt_id`, and output node in Asset meta so a take can be re-run.
7. **Make progress local.** Show queue position, current node, step/total, and
   cancel in the panel; recover the job from `/active_jobs` after reload.
8. **Configure in-app.** Setup should edit URL, enabled, declared GB, optional
   token, and test connection; validate loopback/TLS and warn on remote URLs.
9. **Add a workflow library and template import.** List `object_info` node
   classes, Comfy workflow templates, and saved project workflows with search.
10. **Bound the surface.** Cap JSON size, node count, output bytes, and job
    timeout in the UI and API; show the estimate before queueing.

## Non-goals

- Rebuilding ComfyUI's node-graph editor inside Milimo.
- Downloading ComfyUI models from Milimo.
- Making ComfyUI required for core generation.
- A general-purpose workflow marketplace.

## Implementation plan

The phased plan, acceptance criteria, and effort estimates are in
[[../roadmap/comfyui-frontend-plan-2026-11.md]].
