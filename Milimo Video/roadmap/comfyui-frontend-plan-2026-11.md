---
type: roadmap
created: 2026-11-18
updated: 2026-11-18
tags: comfyui frontend plan workflow roadmap
sources:
  - ../systems/comfyui-frontend-audit-2026-11.md
  - ../../web-app/src/studio/components/ComfyUIPanel.tsx
  - ../../backend/routes/comfy.py
related:
  - ../systems/comfyui-bridge.md
  - open-work-resolution-2026-11.md
  - implementation-plan-2026-11.md
---

# ComfyUI Frontend Implementation Plan (2026-11)

Baseline: generic bridge and paste-JSON panel are shipped. This plan turns it
into a reusable workflow surface without turning Milimo into a node editor.

Priority order: **trust → reuse → library → results → remote → polish**.

## C0 — Trust and recovery (P1, 2–3 days)

| Work | Acceptance |
|---|---|
| Live preflight via `/object_info` | Every `class_type`, required input, enum, and model filename is checked before queueing; missing node fails in Validate, not after submit |
| Per-node error UI | Comfy `node_errors` shown as node/class/message rows with a copy action |
| Job recovery in panel | Reload/remount finds the active comfy job from `/active_jobs`; Cancel and progress still work |
| Per-project draft | Draft key includes `project_id` (and optional shot id); switching projects no longer overwrites the only graph |
| Export/copy/format | Copy JSON, download `.json`, pretty-print, and parse errors include line/column |
| Bounds | Reject >500 nodes, >1 MB JSON, and show a clear limit message |

**Tests:** missing node/model preflight, malformed JSON line, draft isolation,
recovered job cancel, export file.

## C1 — Binding and media bridge (P1, 4–6 days)

| Work | Acceptance |
|---|---|
| Workflow binding schema | `{workflow, bindings: {prompt: "6.text", seed: "3.seed", width: "5.width", height: "5.height", model: "4.ckpt_name"}}`; unbound inputs can still be edited manually |
| Shot binding controls | Prompt, negative, seed, width, height, steps, and selected model are populated from the shot and written into the graph at run time |
| Per-node input panel | A collapsible list of all nodes/inputs from `object_info`; typed widgets for strings, numbers, booleans, enums |
| Model picker | `CheckpointLoaderSimple`, `UNETLoader`, `VAELoader`, etc. list the live Comfy catalogs |
| Media bridge | `POST /comfy/upload` proxies an uploaded/project asset to Comfy `/upload/image` and rewrites bound `LoadImage` nodes; temp inputs cleaned after terminal |
| Reusable graph | The same saved graph runs on multiple shots without editing JSON |

**Tests:** binding writes only declared nodes; missing bound input is a validation
error; uploaded image reaches a live/fake Comfy input; temp cleanup on success,
failure, and cancel.

## C2 — Workflow library and templates (P2, 1–2 weeks)

| Work | Acceptance |
|---|---|
| Persisted workflows | New table/API: project id, name, description, graph JSON, binding map, revision, created/updated; CAS on save |
| Library UI | Create, rename, duplicate, delete, set default; search by name/tag; revision conflict surfaces as a toast |
| Template import | List Comfy `/workflow_templates` and import a template as a draft; report unsupported custom nodes |
| Pack import/export | Export a named workflow pack; import into a project with validation |
| Provenance | Each run records workflow id/hash/version, bindings, Comfy `prompt_id` |

**Tests:** CRUD/revision conflict, template list/import, pack round-trip, provenance
on Asset/Job.

## C3 — Results, progress, and batch (P2, 1 week)

| Work | Acceptance |
|---|---|
| Output result set | Panel lists every output with thumbnail/type/node/size; user selects the primary; “Use as take” is explicit |
| Inline progress | Queue position, current node, step/total, ETA, and cancel are visible in the panel |
| Retry/duplicate | Retry a failed job with the same resolved graph; duplicate guard while a job is active |
| Batch matrix | Queue the same graph across selected shots with prompt/seed bindings; one row per shot |
| Cost/ETA | Resolve declared GB and estimate from the graph; show before queueing; override allowed |

**Tests:** multi-output primary selection, recovered progress, retry, duplicate
guard, batch with bindings, size/ETA display.

## C4 — Remote, setup, and controls (P2–P3, 3–5 days)

| Work | Acceptance |
|---|---|
| In-app config | Setup edits URL, enabled, declared GB, optional bearer token; test connection; persisted settings |
| Remote posture | Warn/refuse non-loopback without explicit opt-in; TLS supported; token never returned to the client |
| Version gate | Minimum Comfy version and missing-endpoint diagnostics |
| Queue controls | Show Comfy queue; cancel/delete only Milimo-owned prompts; clear warning for manual jobs |
| Limits | Server-side node/JSON/output byte caps; output download timeout |

**Tests:** config persistence, remote/token redaction, version mismatch, queue
control isolation, output cap.

## C5 — Editor polish and E2E (P3, 3–4 days)

| Work | Acceptance |
|---|---|
| JSON ergonomics | Line numbers, syntax highlighting, fold/format, search, dirty indicator |
| Accessibility | Keyboard validate/run, focus to first error, labels, axe clean |
| Empty/offline states | “Start ComfyUI”/“Open Setup” affordances; no dead panel |
| E2E | Playwright: import graph, validate, run smoke, select output, delete asset, disabled/offline states |
| Docs | Update `docs/comfyui.md`, user guide, and this plan at each phase |

## Sequencing and dependencies

```text
C0 trust/recovery
  └─ C1 binding/media
       ├─ C2 library/templates
       │    └─ C3 results/batch
       └─ C4 remote/setup
            └─ C5 polish/E2E
```

C0 and C1 are the recommended next release. C2 can start in parallel once the
binding schema is stable; C3 depends on C2 for named workflows; C4 is independent
after C0.

## Definition of done for the track

- A saved graph can be reused across shots with prompt/seed/size/model bindings.
- A reference image can be uploaded from the project and used by `LoadImage`.
- Validate catches missing nodes, missing models, and bad required inputs before
  queueing.
- Workflows survive project switches, reloads, and revisions.
- Multi-output runs have an explicit primary and reproducible provenance.
- Offline, disabled, and remote configurations are explicit and safe.
- No core Milimo path depends on ComfyUI.
