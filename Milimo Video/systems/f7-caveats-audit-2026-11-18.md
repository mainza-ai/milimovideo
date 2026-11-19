---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: frontend ux audit caveats lut relink mobile state contract
sources:
  - Live F8 probe project (`bc3d356e`) with quadrant video, `.cube` and `.3dl` LUTs, one take job
  - web-app/src/studio/components/LutPreview.tsx
  - web-app/src/studio/surfaces/EditSurface.tsx
  - web-app/src/studio/surfaces/LibrarySurface.tsx
  - web-app/src/stores/slices/{projectSlice,shotSlice}.ts
  - backend/routes/{assets,projects,shots}.py
  - backend/utils/render_utils.py
related:
  - frontend-ux-audit-2026-11-18.md
  - ../roadmap/frontend-ux-plan-2026-11-18.md
  - ../roadmap/f7-caveats-plan-2026-11-18.md
  - ../roadmap/implementation-plan-2026-11.md
---

# F7 Caveat Audit — 2026-11-18

## Verdict

The F7 fixes are real, but three caveats were understated and the audit found
**seven new P1 defects** behind them. The most important correction: the WebGL
LUT preview is not merely `.cube`-only — in the normal dev/runtime topology it is
**blocked by cross-origin video taint and renders no usable preview at all.**

Summary by theme:

| Theme | State |
|---|---|
| **LUT preview** | Blocked by CORS; image clips silently skip it; non-cube formats export-only; active-clip only |
| **Relink** | Shot URL rewrites work; job/take provenance, old files, project LUT, and metadata do not |
| **State truth** | Conflict is overwritten by edits; undo after save still says `Saved`; reload during a job says `Pending` |
| **Mobile** | Story is usable; Edit timeline is 121px at 390; drawer has no dialog/focus/Escape semantics |
| **Contracts** | `GET /projects/{id}` → `PUT` the same body returns **500** (`created_at` ISO string) |
| **Verification** | No browser test covers the LUT path, relink provenance, conflict/undo, or the 390px Editor |

## Method

- Built a probe project (`bc3d356e`) with a four-quadrant video (red/green/blue/
  white), a swap-R/B `.cube`, a `.3dl`, one completed take job, and a selected
  take.
- Reproduced the LUT path in Playwright; captured `pageerror`, WebGL state, and
  DOM/canvas presence.
- Ran a same-origin prototype with and without `crossOrigin="anonymous"` to
  isolate the taint.
- Called `/assets/{id}/replace` and compared shot, take, library, and filesystem
  state before/after.
- Ran `GET` → `PUT` round-trip against a fresh project.
- Measured the 390×844 Edit timeline and Story drawer.
- Used two temporary Vitest probes (removed after capturing output) to prove the
  conflict and undo state defects.

## What is working

- `.cube` parser is covered by Vitest (2×2 identity/order test).
- LUT/Titles layering order is correct when the canvas can draw: LUT below
  titles/captions, title overlay visible.
- Relink rewrites `shot.video_url` / `thumbnail_url` for the replaced asset.
- `/uploads?project_id=` is scoped, missing flags work, and bulk selection exists.
- Mobile Story board is 334px at 390 and the drawer opens/closes via the top-bar
  button and backdrop.
- Axe stayed clean during the F7 verification; no new accessibility violation was
  introduced by the caveat paths (the drawer semantics gap is behavioral, not an
  axe failure).
- Backend project rep/relink tests pass; 225 pytest / 102 Vitest / 27 Playwright
  were green at F7 close.

## Resolution — 2026-11-18

All 18 findings were implemented the same day. Original findings remain as the
audit trail.

| Finding | Resolution | Evidence |
|---|---|---|
| F8-01 | Verified root cause and fixed: `crossOrigin`, cache-busted source URL, RGBA8 3D texture, visible 2D canvas, source hidden only after a successful draw | LUT E2E reads TL blue / TR green / BL red / BR white from a quadrant image |
| F8-02 | Relink now migrates Job output/thumbnail, shots, selected take, project LUT, Comfy workflow JSON; preserves `created_at`; reports/optionally purges the old file | `test_relink_migrates_job_take_and_lut_references`, `test_take_replace_relinks_in_place_and_can_purge_old` |
| F8-03 | Conflict is sticky; edits cannot downgrade it; TopBar offers reload or explicit overwrite | `keeps a save conflict sticky through edits` |
| F8-04 | Saved/Unsaved derives from a document signature baseline, so undo/redo are truthful | `marks undo after a save dirty and redo saved again` |
| F8-05 | `created_at` is server-owned and stripped before writes | `test_project_get_put_round_trip` |
| F8-06 | Active jobs hydrate on load and on the global shell hook; card shows spinner + Running | live probe: spinner 1 with a running queued job |
| F8-07 | Mobile full-screen timeline mode; tracks fill the surface; scroll hint; visible thin scrollbar | `ux-matrix` mobile expand test: stage hidden, timeline >400px, 5 tracks |
| F8-08 | Drawer is a dialog with `aria-modal`, focus trap, Escape, focus restore, background inert | `ux-matrix` drawer test asserts Escape and focus return |
| F8-09 | Takes expose `missing`; Takes tab has missing filter, relink, delete | backend project/shot take responses; Library UI |
| F8-10 | LUT preview supports images; image/V2 states are labelled truthfully | LUT E2E uses a still image |
| F8-11 | Parser supports DOMAIN/INPUT_RANGE, rejects 1D and >129 grids; non-cube shows export-only | `LutPreview.test.ts` |
| F8-12 | Relink preserves metadata and project/workflow references; type changes under a take are refused | backend relink tests |
| F8-13 | Library repair runs integrity scan + repair with confirmation | `LibrarySurface.repairMedia` |
| F8-14 | Bulk delete reports partial failures instead of a false total | Library bulk handlers |
| F8-15 | Card status derives from status + media; completed-without-media shows Missing media | `effectiveShotStatus` tests |
| F8-16 | Shared middle-ellipsis util used by Library assets and Comfy outputs | `data/format.ts` |
| F8-17 | Uploads are throttled to frame changes, paused/offscreen, downscaled to 1280 | `LutPreview.tsx` |
| F8-18 | Table view honours hidden tracks; track count computed | `StudioTimeline.tsx` |

Verification: 228 backend, 108 Vitest, 29 Playwright, ruff/tsc/ESLint, build and
bundle green. LUT orientation and channel swap are asserted in `e2e/lut.spec.ts`.

## P1 findings

### F8-01 — LUT preview is blocked by cross-origin video taint

`LutPreview` uploads the `<video>` into a WebGL texture (`texImage2D`,
`LutPreview.tsx:167`). The Edit video element has no `crossOrigin` attribute
(`EditSurface.tsx:298`), while media is served from the API origin (`:8000`) and
the app runs on `:5173`. Chrome rejects the upload:

```text
pageerror: Failed to execute 'texImage2D' on 'WebGL2RenderingContext':
The video element contains cross-origin data, and may not be loaded.
```

A same-origin prototype proves the root cause and the fix path:

```text
without crossOrigin: SecurityError: The canvas has been tainted by cross-origin data
with crossOrigin:    [255, 255, 255, 255]   // readable pixel
```

The API already sends `access-control-allow-origin: http://127.0.0.1:5173`, so a
`crossOrigin="anonymous"` attribute (and the same on the compare video) should
restore the texture upload. Until then the canvas is present but the preview is
black/unusable — the earlier “LUT preview works” claim was wrong.

**Impact:** any project with a LUT shows a broken Edit stage; users cannot judge
the look before export.

### F8-02 — Relink leaves take/job provenance stale and orphans the old file

`replace_asset_file` rewrites only `Shot.video_url` / `Shot.thumbnail_url`
(`assets.py:459-468`). It does not update:

- `Job.output_path` / `Job.thumbnail_path` for takes that used the asset;
- `Shot.selected_take_id` / `favorite_take_id` semantics;
- `project.render_lut` when the relinked asset is the LUT;
- Comfy workflow bindings or `Asset.meta_json` provenance.

Live proof after replacing a video asset with a PNG:

```text
replace 200 /projects/bc3d356e/assets/general/9ebfa8d5_replacement.png image
shot video /projects/bc3d356e/assets/general/9ebfa8d5_replacement.png
take url   /projects/bc3d356e/assets/general/ed83150d_f8.mp4
old==take true   new==take false
```

Consequences:

- `/shots/{id}/takes` still returns the old URL, so the Takes tab points at a file
  the asset no longer owns.
- `/uploads` lists the completed Job as a second, stale card (the old URL is
  still "present" through the job row).
- The old file remains on disk (`ed83150d_f8.mp4`) with no reference — an orphan.
- The shot still selects `take-job-1`, now semantically unrelated to the new
  asset.
- `asset.created_at` is reset to “now” (`assets.py:452`), losing the original
  import date.

### F8-03 — A save conflict is erased by the next edit

`saveProject` sets `saveStatus: 'conflict'` on 409 (`projectSlice.ts:186-187`),
but `updateShot` overwrites any non-runtime edit with `'dirty'`
(`shotSlice.ts:62`). A temporary Vitest probe confirmed:

```text
set conflict → updateShot({prompt}) → saveStatus === 'dirty'
```

The user loses the conflict warning, the TopBar no longer shows the danger tone,
and the next save retries the same stale `revision` — another 409 or, worse, a
merge decision without the conflict context.

### F8-04 — Undo/redo after a save still reports `Saved`

zundo changes `project` directly, bypassing the slice dirty tracking. A temporary
probe saved an edit, pressed undo, and observed:

```text
project.shots[0].prompt !== 'saved edit'   // document changed
saveStatus === 'saved'                     // UI still says Saved
```

The reverse is also stale: undoing back to the saved state stays `dirty`. Save
status must be derived from a document/save baseline, not only from mutations.

### F8-05 — `GET /projects/{id}` → `PUT` the same body returns 500

`_shot_dict` returns `created_at` as an ISO string (`projects.py:182`), but the
PUT sync whitelists `created_at` into `Shot` attributes (`projects.py:442-455`).
SQLModel/SQLite rejects the string:

```text
create shot 200
read created_at 2026-11-18T23:36:10.197142
roundtrip 500 Internal Server Error
StatementError: SQLite DateTime type only accepts Python datetime and date objects
[SQL: UPDATE shot SET keyframes=?, blocking=?, created_at=? ...]
```

This breaks any client that reads then writes a project — integration scripts,
backup/restore, and future SDK users. The Studio save payload omits `created_at`,
so the UI path hides the defect.

### F8-06 — Reload during an active job shows `Pending`

`loadProject` explicitly maps `generating` to `pending` and clears
`isGenerating` (`projectSlice.ts:301-302`). Live probe with a shot row set to
`status='generating'`:

```text
card [ 'Pending' ]   spinner 0
```

`registerActiveJobs` only records job→shot routing; it does not hydrate progress
or status. The UI stays Pending until the next SSE event (if one arrives). A
reload during a long job looks idle and offers Generate again.

### F8-07 — Mobile Edit timeline is 121px tall at 390

Measured with a five-track project:

```text
client 121 / scroll 296
C1 top 885, bottom 939 (viewport 844)
scrollbar auto (overlay, invisible)
```

Only ~2 tracks are visible. The `Tracks 5/5` menu works, but hiding V2 only
reduces scroll to 242. A2/C1 require an undiscoverable scroll gesture. The desktop
five-track fit at 900px does not carry to mobile.

### F8-08 — Mobile Story drawer is not a dialog

```text
role: null   aria-modal: null
Escape → backdrop still present
```

There is no focus trap, no initial focus, and no Escape handling. Keyboard and
screen-reader users can tab into the board behind the drawer; Escape only clears
focus mode.

### F8-09 — Takes have no missing-media state or repair path

`StudioTake` has no `missing` field, `/shots/{id}/takes` returns raw
`job.output_path`, and the Repair action only re-checks Assets. A deleted take
file renders as a normal card and fails only when opened. Relink has no take
counterpart.

### F8-10 — LUT preview is active-clip-only and silently skipped for images

`EditSurface` gates the preview on `!isImage` (`EditSurface.tsx:358`), but the
stage badge still says “preview” whenever `renderLut` exists
(`EditSurface.tsx:395`). Image clips therefore claim a preview while showing none.
With a V2 overlay, the LUT is applied only to the active clip, not the V1/V2
composite the export renders.

## P2 findings

### F8-11 — Non-cube LUTs are export-only; `.cube` parsing ignores domain/range

The render path accepts `.cube`, `.3dl`, `.dat`, `.m3d`
(`projects.py:852-853`); the browser parser only handles `LUT_3D_SIZE` plus
numeric rows. It ignores `DOMAIN_MIN`, `DOMAIN_MAX`, `LUT_3D_INPUT_RANGE`, and 1D
LUTs. Live `.3dl` check: no canvas, badge `LUT on export`, tone `warn` — truthful,
but the feature gap remains.

### F8-12 — Relink metadata and auxiliary references are not migrated

Beyond F8-02: relinking a LUT asset leaves `project.render_lut` pointing at the old
URL; Comfy `LoadImage` bindings and workflow records can point at the replaced
file; `Asset.meta_json` (prompt/seed provenance) is not updated. Relink can also
change the asset type (video → image) while a shot keeps “selected take” status.

### F8-13 — “Refresh & repair” in Library does not repair

`repairMedia` calls `refresh()` and `refreshTakes()` and reports missing counts
(`LibrarySurface.tsx:261-269`). It never calls the real
`POST /projects/{id}/repair` used by Setup. Two different repair stories with
different semantics confuse users.

### F8-14 — Bulk delete hides partial failures

Both bulk paths loop `DELETE /assets/{id}` and count `response.ok`; they do not
surface which ids failed or why, and they do not distinguish 404 from 500. A
partial delete reports “Deleted N” with no recovery list.

### F8-15 — Status and media are not cross-checked

A shot with `status='completed'` but no media shows `Done`; a stale `pending`
shot with a video shows `Pending` on the card while the Generate header counts a
take (because `shotHasTake` also looks at `videoUrl`). The card status should be
derived from the same truth as the take count.

### F8-16 — Filename truncation is Library-asset-only

`middleEllipsis` is applied to Library asset names. Comfy output cards, take
labels, and inspector paths still wrap or use full strings; long generated names
still distort cards.

### F8-17 — LUT preview uploads a full frame every animation frame

`texImage2D` runs on every `requestAnimationFrame` with the native video size
(`LutPreview.tsx:167`). At 1024×576 that is ~2.4 MB per frame, every frame, plus a
draw. There is no pause handling, downscale, or FPS cap. Performance on integrated
GPUs is unmeasured; the risk should be benchmarked after CORS is fixed.

### F8-18 — Track visibility is cosmetic and global

Hiding V2 removes the timeline lane but the Table view still lists it, and the
choice is persisted globally rather than per project. “Tracks 5/5” counts a
hardcoded 5 even when captions are absent.

## Missing functionality / enhancements

1. **A trustworthy LUT preview**: same-origin texture, image support, V2 composite
   awareness, domain/range parsing, and a visible “export-only” state that never
   leaves a black stage.
2. **Provenance-preserving relink**: update jobs, selected/favorite takes, project
   LUT, workflows, and metadata; delete or quarantine the old file only with
   confirmation; keep the original timestamp.
3. **Take-level repair**: missing flags, relink, delete, and filter in the Takes
   tab; one repair surface shared with Setup.
4. **State baseline**: persist `savedProject` (or a hash) and derive `Saved` vs
   `Unsaved` from it, so undo/redo and conflict states are always truthful.
5. **Active-job hydration**: carry `status`, `stage`, `progress`, and ETA for
   running jobs on load, or attach `/active_jobs` to the shot before first paint.
6. **Mobile Edit mode**: a full-screen timeline toggle (or collapsed transport),
   sticky horizontal scrollbar, and a persistent “tracks” affordance.
7. **Dialog-grade drawer**: `role="dialog"`, `aria-modal`, focus trap, Escape,
   restore focus, and background `inert`.
8. **Contract round-trip**: exclude or parse `created_at`; add GET→PUT→GET equality
   tests for projects, shots, captions, and titles.
9. **Partial-failure UX**: per-item results for bulk delete/relink, with retry.
10. **LUT formats**: server-side conversion of `.3dl/.dat/.m3d` to `.cube` for
    preview, or an explicit “export-only” badge plus no canvas.

## Recommended direction

- **Preview is evidence, not decoration.** If the preview cannot match the
  export, say so before the user grades a look. A black canvas is worse than a
  clear “export-only” label.
- **Relink is a reference migration.** Treat it as a graph edit over Asset, Job,
  Shot, Project, and workflow references — not a single URL update.
- **One repair model.** Setup already has scan/repair; Library should call the
  same endpoints and add take-level coverage instead of a second label.
- **Derive save state.** Store the last successfully saved document (or its hash)
  and compare; do not infer “Saved” from the absence of a mutation.
- **Mobile Edit is a mode.** Five tracks cannot fit 121px; offer a full-screen
  timeline instead of pretending the desktop split works.

## Implementation plan

Phased F8 plan with acceptance criteria and effort:
[[../roadmap/f7-caveats-plan-2026-11-18.md]].
