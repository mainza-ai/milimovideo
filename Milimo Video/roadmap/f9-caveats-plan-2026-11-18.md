---
type: roadmap
created: 2026-11-18
updated: 2026-11-18
tags: frontend ux plan caveats jobs relink lut mobile state
sources:
  - ../systems/f9-caveats-audit-2026-11-18.md
  - ../roadmap/f7-caveats-plan-2026-11-18.md
related:
  - ../systems/f9-caveats-audit-2026-11-18.md
  - f7-caveats-plan-2026-11-18.md
  - implementation-plan-2026-11.md
  - open-work-resolution-2026-11.md
---

# F9 Caveat Resolution Plan (2026-11-18)

Input: [[../systems/f9-caveats-audit-2026-11-18.md]] — 21 findings behind the F8
caveats.

Priority: **job truth → destructive-operation safety → preview truth → save
safety → mobile/dialog → verification**.

Status: **implemented — 2026-11-18**. Evidence is in
[[../systems/f9-caveats-audit-2026-11-18]] and the verification gate below.

| Phase | Status | What shipped |
|---|---|---|
| **F9-0** | ✅ | Shell-owned poller (surfaces poll once, no intervals), reconcile on every refresh, vanished jobs clear within one interval, real failed state |
| **F9-1** | ✅ | Project-scoped relink destination, thumbnail purge, shared-reference refusal, metadata invalidation, repair clears take pointers, integrity reports missing takes, take purge/confirmed type change |
| **F9-2** | ✅ | Source error/abort/timeout and context-loss states, 30fps cap, no preserve buffer, compare labelled `LUT · A only`, precision tooltip, video LUT pixel test |
| **F9-3** | ✅ | Delete resets baseline, overwrite snapshots the server copy (localStorage + JSON download), read-only `created_at` in OpenAPI |
| **F9-4** | ✅ | Compact transport in expanded mobile timeline, TopBar/rail inert while drawer open, in-app confirm dialog for repair/purge/overwrite, labeled mobile rail |
| **F9-5** | ✅ | Stale-job/poll-budget E2E, missing-source LUT E2E, video LUT playback E2E, relink/repair backend tests, confirm/delete Vitest |

## Findings → phases

| Phase | Findings | Theme |
|---|---|---|
| **F9-0** | F9-01, F9-02, F9-08 | One job poller and real reconciliation |
| **F9-1** | F9-04, F9-05, F9-06, F9-09, F9-20 | Relink/repair safety |
| **F9-2** | F9-03, F9-11, F9-12, F9-13, F9-14, F9-19 | Preview truth and parity |
| **F9-3** | F9-07, F9-10, F9-18 | Save/delete/overwrite safety |
| **F9-4** | F9-15, F9-16, F9-17 | Mobile transport and dialog primitives |
| **F9-5** | F9-21 | Verification gate |

## F9-0 — One job poller and real reconciliation (P1, 2–3 days)

| Work | Acceptance |
|---|---|
| Single owner | The shell owns `useStudioActiveJobs`; Generate/Comfy/surfaces read the store. One interval, one listener set |
| Reconcile every refresh | After replacing the active list, clear `isGenerating` for shots no longer present; query `/status/{job_id}` for each and apply completed/failed/cancelled |
| Error reconciliation | On a status error, set `isGenerating:false`, `status:'failed'` (or `pending` if unknown), and a visible message — never leave `status:'generating'` |
| Poll budget | No more than one `/active_jobs` request per project per interval when the shell is mounted; test request count |
| Stale job test | Live/E2E: start a fake active job, remove it, assert the card leaves Running within one interval |

**Dependencies:** none. Ship first.

## F9-1 — Relink and repair safety (P1, 4–6 days)

| Work | Acceptance |
|---|---|
| Project-scoped destination | If `asset.path` is missing/legacy, write to `<project>/assets/general`; never emit `/projects/<id>/assets/uploads/...` |
| Purge derived files | Purge removes the old media **and** its old thumbnail; report every removed path |
| Reference check | Before deleting, verify no other Asset/Job/Shot/Project/workflow points at the path; otherwise quarantine or abort |
| Repair lineage | Clearing missing media also clears `selected_take_id`/`favorite_take_id`/`last_job_id` when they point at the missing job; integrity/repair includes missing takes |
| Take relink parity | Take relink gets an explicit purge choice, cleans the old thumbnail, and allows a confirmed cross-kind replacement when no other shot uses the job |
| Metadata | Invalidate `meta_json` provenance on replacement or record both old/new paths |
| Tests | Relink legacy path, thumbnail purge, shared-reference refusal, repair lineage, take purge |

**Dependencies:** F9-0 for status/take truth.

## F9-2 — Preview truth and parity (P2, 4–6 days)

| Work | Acceptance |
|---|---|
| Source failures | Attach `error`/`stalled` listeners; missing image/video, no frame, or blocked source sets `LUT on export` and restores the source visibility |
| Context loss | Handle `webglcontextlost`/`restored`; on loss show the export-only badge and restore the raw media |
| Compare parity | Apply the LUT to Compare B, or label the compare as “A graded / B raw” |
| Precision/color | Document 8-bit/1280px approximation in the badge tooltip; optionally add a float texture path or preview proxy |
| Non-cube formats | Server-side conversion to `.cube` for `.3dl/.dat/.m3d` (preferred), or keep an explicit export-only badge |
| Performance | Benchmark preview FPS on MPS/integrated GPU; avoid the WebGL→2D copy if the direct canvas is reliable, or cap to 24–30fps |
| Cache duplicate | Use a blob URL or a CORS-consistent source instead of a second `?lutPreview=1` download where feasible |
| Tests | Missing source, context loss, compare badge, video playback/seek LUT pixels |

**Dependencies:** F9-0 for shot/media truth.

## F9-3 — Save, delete, and overwrite safety (P2, 2–3 days)

| Work | Acceptance |
|---|---|
| Delete baseline | Deleting the open project resets `savedBaseline`/`saveStatus` to `idle`/`Not saved` for the placeholder |
| Overwrite backup | Before overwriting, snapshot the server document to a timestamped project/JSON export; show a compact diff summary |
| Overwrite scope | Abort if the server changed during the fetch-to-PUT window (revision mismatch) and re-prompt |
| Read-only fields | Mark `created_at` read-only in OpenAPI/Pydantic; document the ignore rule |
| Tests | Delete baseline, overwrite snapshot, revision race, schema read-only |

**Dependencies:** none.

## F9-4 — Mobile transport and dialog primitives (P2, 2–3 days)

| Work | Acceptance |
|---|---|
| Expanded transport | Full-screen timeline keeps a compact play/pause/seek row (and current time), so the cut can be reviewed without collapsing |
| Drawer isolation | While the Story drawer is open, make the TopBar/rail `inert` or `aria-hidden`; restore on close |
| Confirm dialog | Replace `window.confirm` for repair/purge/overwrite with an in-app dialog: title, body, cancel/confirm, danger tone, focus trap |
| Mobile labels | Provide a discoverable label strategy for icon-only rail buttons (long-press/tooltip or bottom sheet) |
| Tests | Expanded transport, drawer background inert, dialog keyboard/axe |

**Dependencies:** F9-1 for purge copy.

## F9-5 — Verification gate (P2, 3 days, continuous)

| Work | Acceptance |
|---|---|
| Job truth | Stale active job clears within one interval; request-count budget test |
| Relink cleanup | Legacy path URL, old thumbnail purge, shared-reference refusal, take purge |
| Preview | Missing source, context loss, compare label, video LUT playback pixels |
| Save | Delete baseline reset, overwrite snapshot, revision race |
| Mobile | Expanded transport, drawer inert, dialog keyboard/axe |
| Docs | Mark F9 phases with commit evidence; close the F8 caveat page gaps |

**Gate:** F9 closes only when backend pytest, Vitest, Playwright, tsc, lint, build,
bundle, and axe are green, plus a live stale-job recovery and a video LUT render
comparison.

## Sequencing

```text
F9-0 job poller
  ├─ F9-1 relink/repair
  ├─ F9-2 preview
  └─ F9-3 save safety
F9-4 mobile/dialog
F9-5 runs from F9-0 onward
```

## Effort

| Phase | Effort | Ships independently |
|---|---|---|
| F9-0 | 2–3 days | Yes |
| F9-1 | 4–6 days | After F9-0 |
| F9-2 | 4–6 days | Yes |
| F9-3 | 2–3 days | Yes |
| F9-4 | 2–3 days | After F9-1 |
| F9-5 | 3 days | Continuous |

Total: roughly **3 working weeks** for one engineer.

## Explicitly not in this plan

- No new render engine or full color-management system.
- No bitmap/LUT authoring UI.
- No replacement of the Job/take ledger.
- No broad mobile navigation redesign beyond transport and labels.

## Definition of done

- A job that disappears never leaves a Running card; one poller serves the shell.
- Relink/repair never leave orphan thumbnails or wrong project URLs, and purge
  respects shared references.
- A LUT preview failure is always visible and never leaves a black/frozen stage.
- Delete resets the baseline; overwrite snapshots the server copy first.
- Mobile full-screen timeline keeps transport, and the drawer isolates the shell.
- All new paths have regression tests plus the F8 suite.
