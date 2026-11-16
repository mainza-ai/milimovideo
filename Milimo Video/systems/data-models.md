---
type: systems
created: 2026-10-09
updated: 2026-10-09
tags: data-models sqlite sqlmodel schemas typescript
sources:
  - docs/02_data_models.md
related:
  - architecture.md
  - frontend-state.md
  - storyboard-engine.md
---

# Data Models

## Purpose

The persistence layer: everything the studio remembers lives in SQLite via
SQLModel/SQLAlchemy (`backend/milimovideo.db`), mirrored by Pydantic schemas on the
API boundary and TypeScript types in the frontend. Six tables, one entity graph.

## Components

**Database.** Engine configured with `check_same_thread=False`, `pool_size=20`,
`max_overflow=40` (survives heavy job polling). On startup, the lifespan handler
marks jobs stuck in `status="processing"` as `"failed"` — zombie recovery so a
crashed worker can't lock the studio.

**Entities** (PKs are `uuid4().hex` strings; `Job.id` looks like `job_XXXXXXXX`):

| Table | Key fields | Relationships |
|---|---|---|
| `project` | name, resolution_w=768, resolution_h=512, fps=25, seed=42, script_content (raw storyboard text) | 1→N shots, scenes, elements, assets |
| `scene` | project_id FK, index, name, script_content (segment) | N→1 project; 1→N shots |
| `shot` | project_id FK, scene_id (nullable), index, track_index (0=V1, 1=V2, 2=A1), start_frame, trim_in/trim_out, action/prompt/dialogue/character, negative_prompt, seed=42, 768×512, num_frames=121, fps=25, enhance_prompt=true, pipeline_override="auto", auto_continue=false, status (pending\|generating\|completed\|failed), last_job_id, video_url, thumbnail_url, matched_elements (JSON), duration=4.0 | N→1 project/scene; 1→N jobs |
| `element` | project_id (indexed), name, trigger_word (e.g. `@Hero`), type (character\|location\|object), description, image_path | N→1 project; trigger words feed IP-Adapter |
| `asset` | project_id (nullable = global), type (image\|video), path, url, filename, width/height, duration, meta_json (prompt, seed, guidance, refs) | N→1 project |
| `job` | project_id (nullable), type (generation\|inpaint), status (pending\|processing\|completed\|failed\|cancelled), progress 0–100, output_path, error_message, prompt, params_json, enhanced_prompt, status_message, actual_frames, thumbnail_path | belongs to project; triggered by shots |

**Pydantic schemas** (`backend/schemas.py`): `ShotConfig` (the full generation
config incl. `timeline: [TimelineItem]` conditioning items), `GenerateAdvancedRequest`
(project_id + ShotConfig), `GenerateImageRequest` (1024×1024 default, 25 steps,
guidance_scale=2.0, dynamic seed, reference_images, enable_ae, enable_true_cfg),
`ElementVisualizeRequest`, `InpaintRequest` (image_path, mask_path, text_mask, points
as strings, prompt), `ScriptParseRequest`, `CommitStoryboardRequest`,
`BatchGenerateRequest`, `BatchThumbnailRequest` (shot_ids, 512×320, force),
`ProjectState`, `ElementCreate`.

**TypeScript types** (`web-app/src/stores/types.ts`) mirror the backend: `Shot`,
`Scene`, `Project`, `StoryElement`, `ConditioningItem`, `MatchedElement`
(element_id, trigger_word, confidence, match_source), plus enums `PipelineOverride`
(auto\|ti2vid\|ic_lora\|keyframe), `ConditioningType`, `ElementType`, `ViewMode`
(timeline\|elements\|storyboard\|images — drives `UISlice.viewMode`).

**On-disk storage** (under `backend/`): `projects/{id}/generated/` (mp4/jpg),
`projects/{id}/thumbnails/`, `workspace/storyboard_artifacts/` (last-frame PNGs for
chained gen), `assets/elements/` (IP-Adapter reference images). Model weights:
`models/flux2/`, `models/sam3/sam3.pt`.

## Key Interactions

- Entity graph: Project → Scene → Shot → Job → output file. Elements attach to shots
  via `matched_elements` JSON (written by `element_matcher.match_elements()`).
- Generation requests carry the full `ShotConfig`; the backend fans out to
  `generate_video_task()` (LTX-2, or Flux 2 when `num_frames==1`, or chained when
  >505 frames) and `generate_image_task()`.
- Thumbnails are Flux 2 image jobs with `is_thumbnail=True`; they write
  `Shot.thumbnail_url` without touching generation state.
- Project save is `PUT /projects/{id}/save` — an upsert that syncs shots and scenes
  (create/update/delete), debounced 2s by `useAutoSave` in the timeline.

## Gotchas

- **Doc inconsistencies (flagged, not resolved):** the Shot entity diagram says
  `cfg_scale` defaults to `3.0` while `ShotConfig` says `2.0`; the entity says
  `upscale` defaults `false` while `ShotConfig` says `True`. Verify against code
  before trusting either.
- **Endpoint naming varies between docs:** `02` maps `POST /generate_advanced` /
  `/generate_image` (underscores); `05`/`06` use `/generate/advanced` /
  `/generate/image` (slashes); the frontend store calls
  `POST /shots/{shotId}/generate`. Treat endpoint paths as approximate until
  checked against `backend/routes/`.
- Shot `status` (pending\|generating\|completed\|failed) and Job `status`
  (pending\|processing\|completed\|failed\|cancelled) are different enums for
  different entities — don't conflate them.
- `InpaintRequest.points` is typed `str`, not a coordinate list.

## Audit Notes (2026-11-16)

Resolved against code; see [[engineering-audit-2026-11.md]]:

- **No project-level shot order exists.** `GET /projects/{id}` returns shots
  `ORDER BY created_at` (`routes/projects.py:94`), while V1 is magnetic in the UI.
  Reordering V1 is therefore lost on reload, and export treats `start_frame` (zero
  for every V1 shot in the live DB) as absolute. Needs a persisted
  `sequence_index` or timeline-item table.
- **`shot.timeline` is an untyped JSON string.** The save path stores whatever the
  client sends (`frameIndex`), the load path reads `frame_index`, and generation
  reads `frame_index`. Conditioning collapses to frame 0 after a round-trip.
- **`Project.updated_at` is never refreshed** (the assignment is commented out at
  `routes/projects.py:199`). There is no revision/ETag for concurrent saves.
- **No migration system.** `create_all()` does not alter existing tables; schema
  changes rely on git-ignored manual scripts. Add Alembic + a startup check.
- **Startup recovery only touches `processing`**, not `pending`; the live DB has
  two stuck `pending` jobs.
- **Defaults are inconsistent:** `Shot.cfg_scale = 1.0`, `ShotConfig = 2.0`,
  frontend `3.0`; `upscale` is `false` (DB), `true` (schema/frontend), `false`
  (task fallback). Pick one and delete the others.
- **Endpoint truth:** `POST /generate/advanced`, `POST /generate/image`,
  `PUT /projects/{id}` (not `/save`); seven routers, not five.

## Related

- [[engineering-audit-2026-11.md]] — persistence findings
- [[architecture.md]] — where these models live in the system
- [[frontend-state.md]] — the TypeScript side of the mirror
- [[storyboard-engine.md]] — scenes, shots, and element matching
