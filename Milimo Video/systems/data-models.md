---
type: systems
created: 2026-10-09
updated: 2026-11-17
tags: data-models sqlite sqlmodel alembic schemas typescript
sources:
  - ../backend/database.py
  - ../backend/schemas.py
  - ../backend/alembic/
related:
  - architecture.md
  - studio-shell.md
  - frontend-state.md
  - storyboard-engine.md
---

# Data Models

## Purpose

The persistence layer: everything the studio remembers lives in SQLite via
SQLModel/SQLAlchemy (`backend/milimovideo.db`), mirrored by Pydantic schemas on the
API boundary and TypeScript types in the frontend. Ten tables, one entity graph,
Alembic-managed.

## Components

**Database.** Engine configured with `check_same_thread=False`, `pool_size=20`,
`max_overflow=40`. Startup runs Alembic to `head`; legacy databases are repaired
with idempotent column adds and stamped at head. A create-all safety net follows for
models introduced between revisions.

**Entities** (PKs are uuid strings unless noted):

| Table | Key fields | Relationships |
|---|---|---|
| `project` | name, resolution_w/h, fps, seed, revision, script_content, board_layout + board_revision, view_state, captions + captions_burn, render_preset, render_crf, render_speed, render_audio_bitrate, render_audio_normalize, storyboard_archive | 1→N shots/scenes |
| `scene` | project_id FK, index, name, script_content | N→1 project; 1→N shots |
| `shot` | project_id, scene_id, index, sequence_index, track_index (0=V1, 1=V2, 2=A1), start_frame, trim_in/out, prompt/negative/action/dialogue/character, seed + seed_locked, width/height/num_frames/fps, cfg_scale, enhance_prompt, upscale, pipeline_override, auto_continue, status, timeline (JSON), keyframes (JSON), blocking (JSON), matched_elements (JSON), selected_take_id, favorite_take_id, audio_gain/muted/fade_in/fade_out, speed, video_url, thumbnail_url, last_job_id | N→1 project/scene; 1→N jobs |
| `element` | project_id, name, trigger_word (`@Hero`), type (character\|location\|object\|style), description, image_path | N→1 project; trigger words feed IP-Adapter |
| `asset` | project_id (nullable), type, path, url, filename, thumbnail_url, width/height, duration, meta_json | N→1 project |
| `queuedjob` | id, job_id, kind (video\|image\|inpaint\|element\|ai_parse\|depth), params_json (incl. `_memory_waits`/`_memory_not_before`), status (queued\|running\|completed\|failed\|cancelled), attempts, resume_policy, timestamps, error_message | scheduler ledger; one row per dispatch |
| `job` | project_id, shot_id, type, status (pending\|processing\|completed\|failed\|cancelled), progress, stage, started_at, completed_at, memory_json, output_path, thumbnail_path, error_message, prompt, params_json, enhanced_prompt, status_message, actual_frames, request_id, resume_policy | user-facing job state + per-stage memory trace |
| `sharelink` | token, project_id, created_at, revoked | capability links for review |
| `reviewitem` | project_id, shot_id, kind (comment\|approval), frame, body, status, author, created_at | frame comments and approvals |
| `appsetting` | key (PK), value_json, updated_at | versioned JSON documents: `llm` (`LLMSettings` v3) and `llm_spend` (daily cloud ledger) |
| `agentmemory` | project_id, scope, shot_id, kind, text, metadata_json, provenance, trust, content_hash, pinned, quarantined, use_count, created/updated/last_used_at | N→1 project; `agentmemory_fts` FTS5 index with insert/update/delete triggers |

**LLM settings document** (`app_setting.key = "llm"`): `schema_version` 3 and a
`revision` CAS token; `providers` keyed by instance id (`ollama`,
`openai_compatible@host:port`) with type, wire, base URL, model, timeout,
keep-alive, declared `memory_gb`, and secret status; `roles` for the six filmmaking jobs with
provider/model/fallback/sampling (empty provider = explicitly unassigned);
`budget` (`daily_usd`, `eval_gate`); and server-written `last_eval`. Credentials
are sealed with AES-256-GCM when `MILIMO_SECRET_KEY` is set and never returned by
reads. `app_setting.key = "llm_spend"` keeps the day's cloud ledger, capped at 200
entries.

**Pydantic schemas** (`backend/schemas.py`): typed `ProjectState`, `ShotState`,
`SceneState`, `CaptionCue`, `TimelineItem`, `BoardLayoutPayload`/`BoardNode`/
`BoardArea`/`BoardLink`, `RenderRequest`/`RenderSettings` (fit, CRF, preset,
audio bitrate, loudness normalize, optional source_path), `ElementCreate`/
`ElementUpdate`, review/share/plan payloads, `DepthRequest`. The LLM plane has its own typed models in
`services/llm/models.py` (`LLMSettings`, `LLMProviderConfig`, `LLMRoleAssignment`,
`LLMBudgetConfig`, `LLMEvalSummary`).

**TypeScript types** (`web-app/src/stores/types.ts` and
`web-app/src/studio/data/useStudioData.ts`) mirror the backend: `Shot` (incl.
`seedLocked`, `matchedElements`, `keyframes`, `blocking`, `timeline`, `stage`),
`Project`
(incl. `renderPreset`, `renderCrf`, `renderSpeed`, `renderAudioBitrate`,
`renderAudioNormalize`), `StudioElement`, `StudioAsset`, takes, review, render, and
agent types. Generated OpenAPI types live in `web-app/src/api/schema.d.ts`; zod
contracts cover health/plans/review/share.

**Migrations:**

| Revision | Purpose |
|---|---|
| `0001_baseline` | Full schema at Alembic adoption; legacy DBs repaired and stamped |
| `0002_render_settings` | `render_crf`, `render_speed`, `render_audio_bitrate`, `render_audio_normalize` |
| `0003_seed_lock` | `shot.seed_locked` |
| `0004_asset_thumbnail` | `asset.thumbnail_url` for uploaded-video posters |
| `0005_app_settings` | `app_setting` table for the revisioned LLM settings document and spend ledger |
| `0006_job_progress` | `job.stage`, `job.started_at`, `job.memory_json` for stage progress and per-job memory traces |
| `0007_shot_priority_steps` … `0011_comfy_workflows` | Shot priority/steps, project LUT, titles, V1 transition, named ComfyUI workflows |
| `0012_agent_memory` | `agentmemory` store plus FTS5 table/triggers (guarded; retrieval falls back to LIKE) |

**On-disk storage** (under `backend/`): `projects/{id}/generated/`,
`projects/{id}/thumbnails/`, `projects/{id}/assets/{general,references,elements}/`,
`projects/{id}/workspace/`, `uploads/`, `models/flux2/`, `models/sam3/`,
`models/depth-anything-v2-small/` (optional).

## Key Interactions

- Entity graph: Project → Scene → Shot → Job → output file. Elements attach to
  shots via `matched_elements` JSON.
- `sequence_index` is the canonical story order. V1 render timing is computed from
  it; `start_frame` is only used by free tracks.
- Project save (`PUT /projects/{id}`) is a revision-checked upsert that syncs shots
  and scenes, serializes typed `timeline`, sanitizes keyframes/blocking/captions,
  and bumps `revision`.
- Board layout saves separately with `board_revision`; view state has its own save.
- Generation carries `ShotConfig`/params through the queue; tasks write `Job` rows
  and SSE events.
- Takes are completed generation jobs; `selected_take_id` and `favorite_take_id`
  point at them without deleting older outputs.
- Integrity scan reports missing media, orphan scene/board references, missing
  sequence indices, empty scenes, and invalid captions. Repair clears broken
  pointers, normalizes order, and sanitizes captions without deleting files.

## Gotchas

- Do not add schema columns without an Alembic revision and a migration test.
- `shot.status` and `job.status` are different enums for different entities.
- JSON columns (`timeline`, `keyframes`, `blocking`, `matched_elements`, captions,
  board layout) are validated/sanitized at the API boundary; do not bypass it.
- `render_*` project fields are defaults for new exports; each render request still
  carries explicit settings.
- `seed_locked` defaults true. Explicit randomize is a user action, not an implicit
  side effect.
- `Job.stage`/`started_at` are written by the throttled progress path and the
  terminal emitter; `/status` returns them so a reload sees the latest stage.
- `Job.memory_json` stores the sampler trace (`peak_gb`, `peak_stage`, per-stage
  peaks/minima, physical minimum, swap peak, samples); `/metrics/jobs` aggregates
  the peak and `/system/memory` returns recent traces.
- Every SQLite connection applies WAL, `synchronous=NORMAL`, a 5 s busy timeout,
  and `foreign_keys=ON`. The bundled 3.45.1 predates the 3.51.3 WAL-reset fix;
  `MILIMO_SQLITE_WAL=0` disables WAL.
- `agentmemory` is project-scoped and deletable; FTS5 is an accelerator, not the
  source of truth, so a build without FTS5 still retrieves via LIKE.
- The LLM document is v3; v1/v2 are migrated on read. `memory_gb` is a declared
  residency override for engines that cannot report it.
- `QueuedJob` is the scheduler truth; `Job` is the user-facing status. Terminal
  states are written once and broadcast once.
- The DB is local and single-writer; no Postgres path is implemented.

## Related

- [[architecture.md]] — where these models live in the system
- [[studio-shell.md]] — how the Studio consumes them
- [[frontend-state.md]] — the TypeScript side of the mirror
- [[storyboard-engine.md]] — scenes, shots, and element matching
- [[../roadmap/implementation-plan-2026-11.md]] — schema and verification status
