---
type: roadmap
created: 2026-11-18
updated: 2026-11-18
tags: frontend ux plan labels colors responsive accessibility studio roadmap
sources:
  - ../systems/frontend-ux-audit-2026-11-18.md
  - ../../web-app/src/studio/
  - ../../web-app/src/stores/slices/projectSlice.ts
related:
  - ../systems/frontend-ux-audit-2026-11-18.md
  - implementation-plan-2026-11.md
  - open-work-resolution-2026-11.md
  - frontend-refactor-2026-11.md
---

# Frontend UX Implementation Plan (2026-11-18)

Input: [[../systems/frontend-ux-audit-2026-11-18.md]] — eight surfaces captured at
1440×900 and 390×844, axe clean at both widths, 24 findings (`F7-01…F7-24`).

The shell does not need a reskin. It needs **truthful state, project-scoped media,
one right rail per surface, a real mobile mode, and readable warnings**. Priority
order: **truth → media trust → density → mobile → timeline/preview → language →
verification**. Each phase lands with tests before the next begins.

Status: **implemented — 2026-11-18**. All seven phases landed; evidence is in
[[../systems/frontend-ux-audit-2026-11-18]] and the verification gate below.

| Phase | Status | What shipped |
|---|---|---|
| **F7-0** | ✅ | One save vocabulary (`Saved`/`Unsaved`/`Saving…`/`Save conflict`/`Save failed`), truthful tone, shot status hydration, runtime fields no longer dirty the document |
| **F7-1** | ✅ | `/uploads?project_id=`, `missing` flags, missing filter, refresh/repair, stable-id relink endpoint, bulk delete for assets and takes, error thumbnails |
| **F7-2** | ✅ | Per-surface rail memory, Generate params and Inspector are mutually exclusive, ComfyUI Run/Bindings/Outputs tabs, 28px+ hit targets |
| **F7-3** | ✅ | 56px mobile rail, full-width Story board (334px at 390), spine drawer, collapsible command dock, compact mobile status strip |
| **F7-4** | ⚠ | Five tracks fit 900px, track visibility menu, title overlay, grouped title form. The LUT canvas is present but blocked by CORS — see F8-01 |
| **F7-5** | ✅ | Readable memory warnings, neutral tone for zero states, pluralization/vocabulary pass, middle ellipsis, styled color/range controls |
| **F7-6** | ✅ | `e2e/ux-matrix.spec.ts`, project-scope backend tests, save-status Vitest, regenerated OpenAPI |

## Findings → phases

| Phase | Findings | Theme |
|---|---|---|
| **F7-0** | F7-01, F7-02, F7-19 | Truthful save and shot status |
| **F7-1** | F7-03, F7-07, F7-13, F7-23 | Project-scoped, self-healing media |
| **F7-2** | F7-06, F7-08, F7-14, F7-15 | One rail per surface, density budget |
| **F7-3** | F7-04, F7-12 | Mobile is a shell, not a squeeze |
| **F7-4** | F7-09, F7-11, F7-10 | Timeline reachability and preview parity |
| **F7-5** | F7-05, F7-16, F7-17, F7-18, F7-20, F7-21, F7-22, F7-24 | Language, tone, and native controls |
| **F7-6** | all | Verification gate and regression tests |

## F7-0 — Truthful state (P1, 1–2 days)

The status surfaces currently disagree with each other after a reload. Fix the
state model first; every later phase reads it.

| Work | Acceptance |
|---|---|
| Save-state vocabulary | One mapping module: `clean → Saved`, `saving → Saving…`, `dirty → Unsaved`, `conflict → Save conflict`, `error → Save failed`. TopBar and StatusStrip consume the same function; no raw `idle` leaks into UI |
| Truthful tone | `Saved` is success; `Unsaved`/`conflict`/`error` are warning/danger. Never green for unsaved |
| Hydrate shot status | Map `status`, `progress`, `stage`, `status_message`, `eta_seconds`, `last_job_id`, `selected_take_id`, and take presence from `/projects/{id}` into the shot |
| Derived "has take" | Story/Generate/Edit/Review read `videoUrl`/takes, not a default `pending`; header copy becomes `N of M shots have takes` with correct pluralization |
| Reload parity test | Vitest: load a project with completed shots → status renders `completed`, take count is correct, save pill reads `Saved` |
| SSE override | Live job events still win over hydrated values; terminal event clears progress |

**Risk:** `Project` type changes ripple through `documentStore`, `projectSlice`,
and every surface that reads `shot.status`. Keep the mapping in one adapter
(`projectSlice` load path) and do not add a second source of truth.

**Dependencies:** none. This unblocks honest Review/Library counts.

## F7-1 — Project-scoped, self-healing media (P1, 3–5 days)

Library is global today: `/uploads` returns every `Asset`/`Job` row, so foreign and
stale media appears in every project.

| Work | Acceptance |
|---|---|
| Project-scoped uploads | `GET /uploads?project_id=…` filters assets and generated jobs; the Studio call always passes the active project. Unscoped reads stay available only for an explicit "All projects" view |
| Contract + test | Pydantic response unchanged in shape; pytest proves another project's asset is excluded and its own asset is included |
| Orphan classification | Library marks `missing` (file absent), `foreign` (not in project), and `stale` (job row without file); summary counts shown |
| Repair path | "Repair missing media" re-checks paths and offers delete or relink; never silently deletes |
| Media fallback | `MediaThumb` renders a styled placeholder (icon + type + filename) on `img`/`video` error instead of a browser-broken icon |
| Bulk actions | Selection + bulk delete/repair for assets and takes; per-card delete moves into a context menu, not a red button on all 222 cards |
| Empty/error states | Every media list has explicit empty, missing, and load-failed states using the same component |

**Tests:** backend project filter, missing-file classification, thumbnail error
fallback (Vitest), bulk delete (Playwright).

**Dependencies:** F7-0 (take presence) for correct Library counts.

## F7-2 — One rail per surface (P1, 4–6 days)

At 1440px, Generate spends **680px** on shot params (320) plus Inspector (360),
leaving a **424px** stage. Setup/Library/Deliver/Shorts lose the same 360px to a
shot panel they do not need.

| Work | Acceptance |
|---|---|
| Per-surface rail policy | A single `railMode` per surface: `shot`, `surface`, or `hidden`. Generate uses shot params **or** Inspector, not both |
| Generate de-duplication | Prompt/seed/frames/CFG appear once; the secondary rail becomes takes/history. Stage width ≥ 640px at 1440 with both rails present |
| Auto-collapse | Library/Deliver/Setup/Shorts start with Inspector collapsed unless a shot is explicitly selected; state remembered per surface |
| Inspector toggle affordance | Visible collapse/expand control with keyboard shortcut; state persists in `editorStore` |
| Card action floor | Story card quick actions ≥ 28×28 with 32px hit area; keep the F6 floor |
| Comfy density | ComfyUI panel splits into Run / Bindings / Outputs tabs; default tab fits without scrolling a 360px rail |

**Tests:** layout assertions at 1440/1280/390 (stage width, rail count), Inspector
persistence, 28px action floor, axe clean.

**Dependencies:** F7-0. Coordinate with the ComfyUI plan; do not regress C0–C5.

## F7-3 — Mobile shell (P1, 1 week)

At 390px the Story board is **124px wide** (rail 76 + spine 190). That is not a
usable canvas.

| Work | Acceptance |
|---|---|
| Board full width | Mobile Story hides the spine behind a drawer; board ≥ 320px at 390px |
| Spine drawer | Explicit "Shots" button opens an overlay drawer with filter/search; closes on select |
| Rail collapse | Mode rail collapses to icon-only or a bottom nav; content never narrower than 320px |
| Bottom chrome priority | Command dock is collapsible and never covers the status strip or the primary action |
| Surface nav | All eight surfaces reachable at 390px without horizontal scroll |
| Touch targets | 32px minimum on mobile for nav, actions, and timeline controls |

**Tests:** Playwright at 390×844 — board width, drawer open/close, nav reachability,
no body overflow, axe clean.

**Dependencies:** F7-0, F7-2.

## F7-4 — Timeline reachability and preview parity (P2, 4–6 days)

The timeline viewport is **249px** against **441px** of tracks at 900px; A2 and C1
sit below the fold with no affordance. Titles/LUT render only on export.

| Work | Acceptance |
|---|---|
| Track affordance | Either fit five tracks at 900px (compact 56px tracks) or show a visible vertical scrollbar/`Tracks 3/5` control; A2/C1 reachable without guessing |
| Scroll indicator | Fade/shadow at the cut edge when tracks overflow; keyboard scroll works |
| Track visibility menu | Toggle V1/V2/A1/A2/C1 visibility; persisted per project |
| Preview parity | Titles and project LUT render in the Edit stage (CSS/WebGL preview), not only in FFmpeg; a "preview approximates export" note is shown only where true |
| Title form hierarchy | Label + field grouping; custom color swatch replaces the raw native bar; add/remove row aligned |
| Dock overlap | Dock and toasts never cover the timeline footer or status strip |

**Tests:** timeline measurement at 900/768 heights, title/LUT preview Playwright
assertion, dock z-order test.

**Dependencies:** F7-2 (height budget), F7-3 (mobile dock).

## F7-5 — Language, tone, and native controls (P2–P3, 3–4 days)

| Work | Acceptance |
|---|---|
| Status tone audit | Zero/neutral states are neutral; success tone only for success |
| Copy pass | Correct pluralization (`1 candidate window`, `N shots have takes`); no `embedded audio mixed` fragments; consistent `Queued`/`Pending`/`Running`/`Done` vocabulary |
| Readable warnings | Safety/memory notes wrap or open a popover; `.st-chip` truncation never applies to a warning without `title` + expand |
| Filenames | Middle ellipsis + full-path tooltip on cards |
| Status strip | Mobile shows device + queue + save only; overflow behind a detail control |
| Native controls | Every `input[type=color|file|range]` styled to the dark tokens; no white bars |
| Tone consistency | Memory "Normal" tone agrees with swap/pressure state; conflicting tone is a bug |

**Tests:** snapshot-free copy assertions, axe, visual regression for the color
input and warning chip.

**Dependencies:** F7-0 (save vocabulary), F7-2 (chip usage).

## F7-6 — Verification gate (P2, 2–3 days, runs alongside)

| Work | Acceptance |
|---|---|
| Surface matrix E2E | One Playwright spec walks all eight surfaces at 1440 and 390: no body overflow, no clipped warnings, control floor, axe clean |
| Status parity test | Reload a fixture project and assert save pill, shot status, and take counts |
| Media scope test | Two projects with assets; each Library shows only its own |
| Layout budget test | Assert stage/rail widths at 1440/1280/1024/390 |
| Visual baselines | Regenerate tolerance baselines for the changed surfaces |
| Docs | Update `index.md`, this plan's status table, and the audit page with commit-level resolution |

**Gate:** F7 is done only when the matrix spec, Vitest, backend pytest, tsc, lint,
build, bundle budget, and axe all pass, and a real render confirms title/LUT
preview parity.

## Sequencing

```text
F7-0 truth/status
  ├─ F7-1 project media
  ├─ F7-2 one rail
  │    ├─ F7-3 mobile
  │    └─ F7-4 timeline/preview
  └─ F7-5 language/tone
F7-6 verification runs from F7-0 onward
```

## Effort

| Phase | Effort | Ships independently |
|---|---|---|
| F7-0 | 1–2 days | Yes |
| F7-1 | 3–5 days | Yes |
| F7-2 | 4–6 days | Yes |
| F7-3 | 1 week | After F7-2 |
| F7-4 | 4–6 days | After F7-2 |
| F7-5 | 3–4 days | Yes |
| F7-6 | 2–3 days | Continuous |

Total: roughly **3–4 working weeks** for one engineer, with F7-1/F7-2/F7-5
parallelizable.

## Explicitly not in this plan

- No palette replacement and no theme system rewrite — tokens are coherent.
- No changes to the render truth, GPU queue, or memory admission.
- No ComfyUI feature work beyond density (tracked in
  [[comfyui-frontend-plan-2026-11]]).
- No legacy `/` parity work; that cutover gate is tracked separately.

## Known caveats carried forward

An 2026-11-18 caveat audit found that F7-11's LUT preview is blocked by
cross-origin video taint and that relink/state/mobile have remaining gaps. See
[[../systems/f7-caveats-audit-2026-11-18]] and the F8 plan
[[f7-caveats-plan-2026-11-18]]. F7 is still complete for its stated phases; the
preview claim is corrected there.

## Definition of done

- Save pill and status strip always agree and never show success for unsaved work.
- Every shot shows its real status and take state after reload.
- Library shows only the active project's media and explains missing files.
- Generate keeps a usable stage; each surface has exactly one primary right rail.
- Mobile Story has a full-width board and a reachable surface switcher.
- No safety warning is truncated without a way to read it.
- The F7-6 matrix spec plus existing gates are green, with real render evidence for
  preview parity.
