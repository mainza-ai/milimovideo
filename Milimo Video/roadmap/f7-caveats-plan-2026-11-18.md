---
type: roadmap
created: 2026-11-18
updated: 2026-11-18
tags: frontend ux plan caveats lut relink mobile state contract
sources:
  - ../systems/f7-caveats-audit-2026-11-18.md
  - ../roadmap/frontend-ux-plan-2026-11-18.md
  - ../../web-app/src/studio/components/LutPreview.tsx
  - ../../backend/routes/assets.py
related:
  - ../systems/f7-caveats-audit-2026-11-18.md
  - frontend-ux-plan-2026-11-18.md
  - implementation-plan-2026-11.md
  - open-work-resolution-2026-11.md
---

# F7 Caveat Resolution Plan (2026-11-18)

Input: [[../systems/f7-caveats-audit-2026-11-18.md]] — 18 findings (`F8-01…F8-18`)
behind the caveats reported at F7 close.

Priority order: **contract truth → state truth → preview truth → provenance →
mobile → verification**. The contract and state phases are small and unblock
trustworthy testing of the rest.

Status: **implemented — 2026-11-18; caveats re-audited in F9**. Evidence is in
[[../systems/f7-caveats-audit-2026-11-18]] and the verification gate below.

| Phase | Status | What shipped |
|---|---|---|
| **F8-0** | ✅ | `created_at` stripped from writes; GET→PUT→GET round-trip test |
| **F8-1** | ✅ | Baseline-derived Saved/Unsaved, sticky conflict with reload/overwrite, active-job hydration on boot, status/media cross-check |
| **F8-2** | ✅ | CORS-capable LUT source with cache-busting, RGBA8 3D texture, 2D visible surface, domain/range parsing, image support, active-clip badge, throttled uploads |
| **F8-3** | ✅ | Relink migrates jobs/takes/LUT/workflows, preserves timestamps, optional purge with explicit confirm; take replace endpoint and missing flags; real Library repair; per-item bulk errors |
| **F8-4** | ✅ | Mobile full-screen timeline, scroll hint, dialog drawer with focus trap/Escape, shared filename ellipsis, table respects hidden tracks |
| **F8-5** | ✅ | LUT pixel E2E, mobile drawer/timeline E2E, relink provenance tests, contract round-trip test, conflict/undo Vitest |

## Findings → phases

| Phase | Findings | Theme |
|---|---|---|
| **F8-0** | F8-05 | Contract round-trip |
| **F8-1** | F8-03, F8-04, F8-06, F8-15 | Save/shot truth |
| **F8-2** | F8-01, F8-10, F8-11, F8-17 | LUT preview truth |
| **F8-3** | F8-02, F8-09, F8-12, F8-13, F8-14 | Relink and repair |
| **F8-4** | F8-07, F8-08, F8-16, F8-18 | Mobile and residual polish |
| **F8-5** | all | Verification gate |

## F8-0 — Contract round-trip (P1, 1–2 days)

`GET /projects/{id}` returns `shot.created_at` as an ISO string; `PUT` accepts it
and SQLModel rejects it. Any read-modify-write client gets a 500.

| Work | Acceptance |
|---|---|
| Exclude server-owned fields | `created_at` (and any other server-owned timestamps/ids) are removed before `setattr`/`Shot(**clean_data)`; client-sent values cannot overwrite creation time |
| Accept legacy payloads | A POST body containing `created_at` is ignored, not rejected with 500 |
| Round-trip test | `GET /projects/{id}` → `PUT` the exact body → `200`, then `GET` again and compare shots/scenes/captions/titles/revisions |
| OpenAPI note | `ShotState.created_at` documented as read-only/ignored on write |

**Tests:** backend round-trip for a project with shots, captions, titles, a board,
and a completed take; also an empty project.

**Dependencies:** none. Ship first.

## F8-1 — Save and shot truth (P1, 3–4 days)

| Work | Acceptance |
|---|---|
| Conflict is sticky | Once `saveStatus = conflict`, edits keep the conflict state (or move to a distinct `conflict-dirty`); only a successful reload/merge clears it. TopBar stays danger-toned |
| Save baseline | Store the last saved document hash (or snapshot); derive `Saved`/`Unsaved` from `project !== baseline`, so **undo/redo** and `setProject` are correct without per-mutation flags |
| Conflict recovery | Offer “Reload server copy” and “Overwrite with mine” explicitly; overwrite sends the server revision, not the stale local one |
| Active-job hydration | On load, fetch `/projects/{id}/active_jobs` before painting shot status; map `generating` to `Running` with stage/progress/ETA. If the job is gone, reconcile from `/status/{job_id}` |
| Status/media cross-check | Card status derives from `status` + media presence: completed without media → `Missing media` (warn); pending with media → `Done`; header take count uses the same helper |
| Tests | Vitest: conflict + edit, undo/redo after save, runtime fields, status/media matrix. Playwright: reload with a running job shows Running |

**Dependencies:** F8-0 (round-trip test harness).

## F8-2 — LUT preview truth (P1, 4–6 days)

| Work | Acceptance |
|---|---|
| Fix texture upload | Edit/compare videos load with `crossOrigin="anonymous"` (API already sends ACAO); WebGL texture upload succeeds in dev and desktop |
| No black stage | If the LUT cannot be applied, the raw video stays visible with an explicit `LUT export-only` badge. A canvas may never cover the stage without a successful draw |
| Image clips | LUT preview renders over still images too, or the badge says `LUT export-only` — never “preview” when none is applied |
| Composite parity | Preview either applies the LUT to the V1/V2 composite or labels itself “active clip only” when a V2 overlay is present |
| Parser coverage | Parse `DOMAIN_MIN`/`DOMAIN_MAX` and `LUT_3D_INPUT_RANGE`; reject 1D LUTs explicitly. Keep the 2×2 order test |
| Non-cube formats | Either convert `.3dl/.dat/.m3d` to `.cube` server-side for preview or show a clear export-only state. No silent failure |
| Performance | Throttle texture uploads to the video frame rate, pause when paused/offscreen, and downscale large frames. Add a measured FPS check on MPS |
| Orientation | A quadrant-video pixel test proves no horizontal/vertical mirror after the CORS fix |

**Tests:** Vitest parser cases (domain/range, 1D rejection); Playwright LUT path
with a quadrant clip asserting canvas pixels or a visible export-only badge;
manual MPS FPS measurement.

**Dependencies:** F8-0 for any API conversion endpoint.

## F8-3 — Relink and repair (P1, 5–7 days)

| Work | Acceptance |
|---|---|
| Reference migration | Relink updates `Asset`, every `Job.output_path`/`thumbnail_path`, shot `video_url`/`thumbnail_url`, and `selected_take_id`/`favorite_take_id` when they belong to the replaced asset |
| Project and workflow refs | Update `project.render_lut` when the relinked asset is the LUT; update or flag Comfy workflow bindings that referenced the old URL |
| Old file handling | Keep the original timestamp; delete or quarantine the old file only after explicit confirmation; report the old path in the response |
| Take repair | `/shots/{id}/takes` includes `missing`; Takes tab gets missing filter, relink, and delete; repair covers takes |
| One repair model | Library “Refresh & repair” calls `GET /integrity` + `POST /repair` (same as Setup); the label describes what actually happened |
| Bulk failures | Bulk delete/relink returns per-item success/failure and shows a retry list; no “Deleted N” when some failed |
| Metadata | Preserve `meta_json` or mark provenance stale; never silently change an asset’s type under a selected take |
| Tests | Backend: relink migrates jobs/takes/LUT refs and reports the old path; take missing flags. Playwright: relink then open Takes and the shot |

**Dependencies:** F8-1 (status/take truth), F8-2 (LUT reference path).

## F8-4 — Mobile editing and polish (P2, 4–6 days)

| Work | Acceptance |
|---|---|
| Full-screen timeline | Mobile Edit gets an explicit Timeline mode (or collapsible transport) that gives the track area ≥ 50% of the viewport; A2/C1 reachable without a hidden gesture |
| Sticky affordance | Visible horizontal/vertical scroll indicator or `3/5 tracks` control; track menu stays reachable |
| Drawer semantics | Story drawer is `role="dialog"` + `aria-modal`, traps focus, closes on Escape, restores focus, and makes the background inert |
| Filenames | One truncation policy across Library assets, takes, Comfy outputs, and inspector paths; full value in `title` |
| Track visibility | Table view honors hidden tracks or the menu states “timeline visibility only”; persist per project |
| Tests | Playwright 390: timeline height/visibility, Escape closes drawer, focus returns, axe clean |

**Dependencies:** F8-1/F8-3 for take labels and status.

## F8-5 — Verification gate (P2, 2–3 days, continuous)

| Work | Acceptance |
|---|---|
| Contract test | GET→PUT→GET equality for a populated project |
| Relink provenance test | Jobs, takes, selected/favorite ids, project LUT, old path reported |
| State tests | Conflict sticky, undo/redo truth, active-job hydration, status/media matrix |
| LUT runtime test | Cross-origin video + `.cube` canvas draws; non-cube badge; image clip badge; orientation quadrants |
| Mobile test | 390×844 Edit timeline and drawer behavior |
| Docs | Mark F8 phases done with commit evidence; update the F7 caveat section to point here |

**Gate:** F8 closes only when backend pytest, Vitest, Playwright, tsc, lint, build,
bundle, and axe are green, plus the LUT pixel/orientation check and a real export
comparison.

## Sequencing

```text
F8-0 contract
  └─ F8-1 state truth
       ├─ F8-2 LUT preview
       │    └─ F8-3 relink/repair
       └─ F8-4 mobile
F8-5 runs from F8-0 onward
```

## Effort

| Phase | Effort | Ships independently |
|---|---|---|
| F8-0 | 1–2 days | Yes |
| F8-1 | 3–4 days | Yes |
| F8-2 | 4–6 days | Yes |
| F8-3 | 5–7 days | After F8-1/F8-2 |
| F8-4 | 4–6 days | After F8-1 |
| F8-5 | 2–3 days | Continuous |

Total: roughly **3–4 working weeks** for one engineer.

## Explicitly not in this plan

- No new LUT authoring/editing UI; only truthful preview and format coverage.
- No replacement of the Job/take ledger with a new table.
- No broad mobile redesign beyond Edit timeline and drawer semantics.
- No changes to render truth, the GPU queue, or memory admission.

## Known caveats carried forward

The F9 audit [[../systems/f9-caveats-audit-2026-11-18]] found stale active-job
cards, silent LUT source failures, thumbnail/relink cleanup gaps, repair/delete
state, and mobile transport issues. The F9 plan fixed all of them:
[[f9-caveats-plan-2026-11-18]].

## Definition of done

- A LUT never produces a black stage; preview, export-only, and image cases are
  visibly distinct and pixel-verified.
- Relink is a full reference migration that reports the old file and leaves no
  stale take/job/LUT pointers.
- Saved/Unsaved/Conflict reflect the actual document baseline through edits,
  undo, redo, and reload.
- A reload during a running job shows Running with progress.
- Mobile Edit can reach every track without a hidden gesture, and the drawer
  behaves like a dialog.
- A populated project round-trips through GET→PUT with no 500.
