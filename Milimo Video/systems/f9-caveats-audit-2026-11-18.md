---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: frontend ux audit caveats jobs relink lut mobile state
sources:
  - Live F9 probes (stale active job, missing LUT source, request polling)
  - Temporary relink thumbnail probe
  - web-app/src/studio/data/useStudioData.ts
  - web-app/src/studio/events/jobBridge.ts
  - web-app/src/stores/slices/projectSlice.ts
  - web-app/src/studio/state/documentStore.ts
  - backend/routes/assets.py
  - backend/routes/shots.py
  - backend/routes/projects.py
related:
  - f7-caveats-audit-2026-11-18.md
  - ../roadmap/f8-caveats-plan-2026-11-18.md
  - ../roadmap/f9-caveats-plan-2026-11-18.md
  - ../roadmap/implementation-plan-2026-11.md
---

# F8 Caveat Audit — 2026-11-18

## Verdict

F8 fixed the reported defects, but the fixes left **four trust gaps** that show up
in normal use:

1. **A running job can stay “Running” forever** after it disappears from
   `/active_jobs`; the 2.5s poll refreshes the list but never reconciles shots.
2. **LUT preview failures are silent** when the media itself fails to load — no
   canvas, no error, and the badge still says `preview`.
3. **Relink purge leaves old thumbnails behind** and can write a wrong project URL
   for assets whose `path` is empty/legacy.
4. **Repair and delete leave stale lineage/state**: selected take ids survive a
   media repair, and deleting the open project leaves the previous save baseline.

Also found: duplicate active-job pollers, a destructive overwrite with no backup,
LUT/compare and precision mismatches, mobile transport loss in full-screen mode,
and several test gaps.

## Method

- Read the F8 implementations end to end, including cleanup paths.
- Live probe: a project with a running job in `/active_jobs`, then deleted the job
  rows and waited 6s. Counted `/active_jobs` requests for ~9.5s.
- Live probe: a project with `render_lut` set and a missing media URL.
- Temporary pytest: upload a video with a generated poster, relink with
  `purge_old=true`, then check the old thumbnail.
- Reviewed save/conflict, repair, delete, mobile, and LUT error paths against the
  F8 acceptance criteria.

## What is working

- LUT `.cube` preview draws correctly for video and stills (F8 pixel test:
  TL blue, TR green, BL red, BR white under a swap LUT).
- Relink migrates job/take URLs, shot URLs, project LUT, and Comfy workflow JSON.
- Save conflict is sticky; undo/redo truth is derived from the saved baseline.
- Active jobs hydrate on boot and show a spinner + `Running`.
- Mobile full-screen timeline and dialog drawer work; axe remains clean.
- Contract round-trip, relink provenance, and LUT pixel tests are green.

## Resolution — 2026-11-18

All F9 findings were implemented the same day. Original findings remain as the
audit trail.

| Finding | Resolution | Evidence |
|---|---|---|
| F9-01 | `refresh()` reconciles every poll; vanished jobs complete/fail within one interval | `e2e/jobs.spec.ts`: Running → Done |
| F9-02 | Shell owns the interval; Generate/Comfy use `poll:false` | request budget ≤5 in the stale-job test |
| F9-03 | Source `error`/`abort`/6s no-frame and context loss set export-only | `e2e/lut.spec.ts` missing-source test |
| F9-04 | Purge removes media + thumbnail; shared refs are refused | relink thumbnail/shared tests |
| F9-05 | Project-scoped destination/URL helper; legacy empty path test | `test_relink_legacy_empty_path_uses_project_assets` |
| F9-06 | Repair clears take pointers and reports missing takes | `test_repair_clears_take_pointers_for_missing_video` |
| F9-07 | Delete resets baseline/status to idle | `saveStatus.test.ts` delete test |
| F9-08 | Reconcile error sets `failed`, not `generating` | hook refactor |
| F9-09 | `_other_asset_refs` blocks purge of shared paths | shared-reference test |
| F9-10 | Overwrite snapshots server JSON (localStorage + download) before PUT | `overwriteProject` |
| F9-11 | Compare badge reads `LUT · A only` | EditSurface |
| F9-12 | 8-bit/1280px documented in the preview tooltip | EditSurface |
| F9-13 | WebGL context-loss handler restores the source + badge | LutPreview |
| F9-14 | Non-cube formats keep explicit `LUT on export` | badge |
| F9-15 | Expanded mobile timeline keeps play/seek transport | `ux-matrix` test |
| F9-16 | TopBar/rail/status/dock inert while drawer open | `ux-matrix` inert assertion |
| F9-17 | In-app confirm dialog replaces repair/purge/overwrite confirms | `ConfirmDialog` + store test |
| F9-18 | `created_at` is readOnly in OpenAPI | schema test |
| F9-19 | Uploads capped at 30fps, preserve buffer removed | 1024×576 headless: **27.3fps** |
| F9-20 | Take relink has purge/type confirmation and cleans thumbnails | take replace test |
| F9-21 | Tests added for all major paths | 232 backend / 110 Vitest / 32 Playwright |

## P1 findings

### F9-01 — A disappeared active job leaves a permanent “Running” card

`useStudioActiveJobs` refreshes `/active_jobs` every 2.5s but only calls
`reconcile()` on mount and on `resync`/`job-terminal` (`useStudioData.ts:775-781`).
`refresh()` replaces the job list and `registerActiveJobs()` only *adds* shot
state; it never clears a shot missing from the new list.

Live probe:

```text
before delete: { status: 'Running', spinner: 1 }
delete queuedjob + job, set shot completed
after 6s:      { status: 'Running', spinner: 1 }
```

If the terminal event is lost (tab was asleep, SSE dropped, job cancelled from the
CLI), the card never recovers. The shot stays `generating`, Generate stays
disabled, and the user cannot start a new take without a full reload.

### F9-02 — Multiple owners poll `/active_jobs`

The shell calls `useStudioActiveJobs(project.id)` (`StudioApp.tsx:71`), Generate
does the same (`GenerateSurface.tsx:56`), and the Comfy panel adds another
(`ComfyUIPanel.tsx:452`). Each hook owns a 2.5s interval, event listeners, and
store writes.

Live count: **7 requests in ~9.5s** from a shell-only probe (2 pollers). With
Generate and Comfy mounted it can reach 3× the intended rate. This is avoidable
traffic and repeated shot updates that can race terminal events.

### F9-03 — LUT preview fails silently when the media does not load

`LutPreview` only calls `onError` from its fetch/WebGL catch. It waits for
`sourceReady()` forever if the image/video itself 404s, has no frame, or is
blocked; no source `error` listener sets `failed`.

Live probe with a valid `.cube` and a missing video URL:

```text
canvas: 0
badges: ['preview']
stage empty: false
```

The stage shows a broken media element, the badge claims a preview, and there is
no `LUT on export` warning. The user cannot tell whether the look is missing or
the media is missing.

### F9-04 — Relink purge leaves old thumbnails behind

`replace_asset_file` captures `old_thumbnail` for reference migration but purges
only `old_path` (`assets.py:524-530`). The temporary pytest uploaded a video with a
generated `*_thumb.jpg`, relinked with `purge_old=true`, and confirmed the old
thumbnail still exists. Take relink has the same gap and never captures the old
thumbnail at all (`shots.py:215-221`).

Repeated relinks accumulate poster images that no row references.

### F9-05 — Relinking a legacy/empty-path asset writes a wrong URL

The new folder and URL are inferred from `asset.path`:

```python
folder = os.path.dirname(asset.path) if asset.path else UPLOAD_DIR
...
folder_name = os.path.basename(folder)
asset.url = f"/projects/{asset.project_id}/assets/{folder_name}/{filename}"
```

If `path` is empty or points at legacy `backend/uploads`, the file is written to
`uploads/` but the URL becomes `/projects/<id>/assets/uploads/<file>` — a path
that does not exist. The Library then shows the relinked card as missing again.
Assets imported before project-scoped storage are the likely victims.

### F9-06 — Repair clears media but keeps the take lineage

`repair_project` nulls `video_url`/`thumbnail_url` for missing files
(`projects.py:678-688`) but leaves `selected_take_id`, `favorite_take_id`, and
`last_job_id`. The shot then renders “Missing media” while `/shots/{id}/takes`
still reports a selected take pointing at the broken job. Library repair also
ignores missing **take** files; Setup integrity and the take tab disagree about
what is broken.

### F9-07 — Deleting the open project leaves the old save baseline

`deleteProject` replaces the document with a fresh placeholder
(`projectSlice.ts:446-449`) but does not reset `savedBaseline` or `saveStatus`.
The store subscription compares the new placeholder with the deleted project’s
baseline and marks it dirty, so a brand-new project opens as **Unsaved** (or, in
rare cases, inherits a conflict).

### F9-08 — Reconcile errors leave a split state

When `/status/{job_id}` fails, `reconcile` sets:

```ts
{ isGenerating: false, statusMessage: 'Unknown', stage: 'failed' }
```

It does **not** set `status`, so `status` stays `generating`. The card stops
spinning but still reads `Running`; `effectiveShotStatus` also returns
`generating`, and Generate keeps the shot disabled.

### F9-09 — Purge does not check other references to the old file

`purge_old` deletes `old_path` after updating Asset, Job, Shot, Project, and
workflow rows, but it does not ask whether another Asset/Shot/Job still points at
the same path (duplicate imports, copied assets, a take and an upload sharing a
file). Purging can therefore break an unrelated card that was not in the update
set.

### F9-10 — Conflict overwrite is a silent full replace

`overwriteProject` fetches the server revision and PUTs the whole local document
(`projectSlice.ts:416-430`). There is no backup, diff, or export before the
server copy is replaced. The user confirms with a browser confirm, but one
misclick discards every remote change made since the conflict.

## P2 findings

### F9-11 — Compare B is not LUT-graded

The LUT canvas wraps only the A video/image. The compare `<video>`/`<img>` is raw.
Switching to Side/Wipe compares a graded A against an ungraded B, which makes the
grade look wrong and can mislead a take decision.

### F9-12 — Preview precision and color management differ from export

The preview uses an **8-bit RGBA8** 3D texture, downscales source frames to
1280px, and samples through the browser canvas color space. Export uses FFmpeg
`lut3d` trilinear interpolation at full resolution with the render pipeline's
color handling. Banding, highlight roll-off, and high-frequency detail can
differ even when the LUT is identical.

### F9-13 — No WebGL context-loss handling

`LutPreview` has no `webglcontextlost`/`restored` listeners. If the GPU context is
lost after the first draw, the canvas freezes (or clears) while the source is
`visibility:hidden`; no badge appears and the user cannot recover except by
reopening Edit.

### F9-14 — Non-cube LUTs still have no preview

`.3dl`, `.dat`, and `.m3d` are accepted for export but show `LUT on export`. This
is truthful, but a server-side conversion to `.cube` (or a parser for `.3dl`)
would close the feature gap.

### F9-15 — Full-screen mobile timeline hides the transport

The expanded CSS hides `.st-stage` and `.st-transport` (`studio.css:3507-3510`).
There is no play/pause/seek control inside the timeline; users must collapse the
timeline to review a cut. Spacebar still works, but that is not discoverable on a
phone.

### F9-16 — Drawer is modal to keyboard focus, not the whole shell

The drawer sets `role="dialog"`/`aria-modal`, traps Tab, and makes the board
`inert`. The TopBar and mode rail remain interactive and exposed to a screen
reader's virtual cursor. Either set the rest of the shell inert or scope
`aria-hidden` while open.

### F9-17 — Blocking browser confirmations

Repair, asset purge, and conflict overwrite use `window.confirm`. This breaks the
dark visual system, cannot be styled/keyboard-controlled consistently, and is not
tested. A small in-app confirm dialog would match the rest of Studio.

### F9-18 — `created_at` is ignored but still advertised as writable

The server now strips `created_at`, but `ShotState.created_at` remains a generic
writable field in OpenAPI (`schema.d.ts` shows `unknown`). Clients can send it and
silently lose it. Mark it read-only or document the ignore rule.

### F9-19 — LUT preview duplicates media cache and copies every frame

The cache-busting `?lutPreview=1` creates a second browser cache entry for the
same video/image. Every drawn frame also does `gl.drawArrays` → `drawImage` into
a visible 2D canvas with `preserveDrawingBuffer: true`. The cost is bounded by the
1280px downscale and frame-change throttle, but no MPS/integrated-GPU FPS
benchmark exists.

### F9-20 — Take relink always keeps the old file and blocks cross-kind repair

The take UI has no purge choice (unlike assets) and the endpoint rejects
image↔video replacement even when the take is missing and the user explicitly
wants to replace it. Old take thumbnails are never cleaned.

### F9-21 — Verification gaps

No automated coverage for:

- video LUT playback/seek (only a still-image LUT test),
- missing LUT source (the silent-failure path),
- WebGL context loss,
- stale/disappearing active jobs,
- old-thumbnail purge,
- repair leaving selected take ids,
- delete-project baseline reset,
- expanded mobile timeline transport,
- duplicate poller request rate.

## Missing functionality / enhancements

1. **One job poller** owned by the shell, with reconciliation on every refresh and
   a single source of truth for active/queued jobs.
2. **Truthful preview failure states**: source `error`, `stalled`, context loss,
   and no-frame cases must surface `LUT on export` instead of `preview`.
3. **Grade comparison**: apply the LUT to Compare B, or label compare as
   “A graded / B raw”.
4. **Reference-safe purge**: check duplicate references, remove old thumbnails,
   and quarantine instead of delete by default.
5. **Project-scoped relink**: always write into the project's `assets/general`
   when `path` is missing/legacy; never emit an `/assets/uploads/` URL.
6. **Repair/take coordination**: clear selected/favorite/last job pointers when
   media is removed, and include missing takes in integrity/repair.
7. **Save safety**: reset baseline on delete; snapshot the server copy before
   overwrite; offer diff/export.
8. **Mobile transport**: a compact play/seek row in expanded timeline mode.
9. **Dialog primitives**: replace `window.confirm` with an in-app confirm.
10. **LUT parity**: consider a server-side preview proxy for non-cube formats and
    document precision/color differences.

## Recommended direction

- **One owner for background truth.** The shell should own active-job polling;
  surfaces read the store. Reconcile every refresh, not only on reconnect.
- **Preview failure is a state, not an absence.** A missing canvas must never look
  like a successful preview.
- **Cleanup follows references.** Purge only when no other row uses the path; clean
  thumbnails and derived files with the media.
- **Delete/repair/overwrite are destructive operations.** Snapshot or confirm with
  a real dialog, and clear dependent ids explicitly.
- **Mobile expanded mode is for editing the cut**, so it needs transport, not just
  tracks.

## Implementation plan

Phased F9 plan with acceptance criteria and effort:
[[../roadmap/f9-caveats-plan-2026-11-18.md]].
