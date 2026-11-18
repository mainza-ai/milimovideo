---
type: systems
created: 2026-11-18
updated: 2026-11-18
tags: frontend ux audit labels colors responsive accessibility studio
sources:
  - Live Studio probe project (7 shots, 1440×900 and 390×844)
  - web-app/src/studio/
  - web-app/src/studio/studio.css
  - web-app/src/stores/slices/projectSlice.ts
related:
  - studio-ux-audit-2026-11.md
  - studio-shell.md
  - ../roadmap/frontend-ux-plan-2026-11-18.md
  - ../roadmap/implementation-plan-2026-11.md
---

# Frontend UX Audit — 2026-11-18

## Verdict

The shell is visually coherent and accessible, but several **truthfulness and
density regressions** remain. It is not a color-palette problem: axe reports zero
serious/critical violations on all eight surfaces at desktop and mobile widths.
The real issues are state that lies after reload, project-global media, two
competing right rails, a mobile Story layout with almost no canvas, and safety
warnings that are truncated.

**No new P0 visual/accessibility defect. Six P1 product/UX defects.**

## Method

- Created a live probe project with seven shots (V1, V2, A1), posters, videos,
  captions, and a title.
- Captured all eight surfaces at **1440×900** and **390×844**.
- Ran axe (`wcag2a`, `wcag2aa`) on every surface at both widths.
- Measured body overflow, clipped text, text under 10px, buttons under 28px,
  rail/spine/board widths, Inspector/stage/params widths, and timeline scroll.
- Reviewed `projectSlice.ts`, `TopBar.tsx`, `StatusStrip.tsx`, `MediaThumb.tsx`,
  `MemoryPanel.tsx`, `StudioTimeline.tsx`, and `studio.css`.

## What is working

- **Axe:** 0 serious/critical violations on Story, Generate, Edit, Review,
  Library, Deliver, Shorts, and Setup at both 1440 and 390.
- **No body overflow:** `scrollWidth === innerWidth` on every measured surface.
- **Typography/control floor:** 0 text nodes under 10px and 0 buttons under 28px
  on every surface except four Story card quick actions (26×26).
- Semantic track colors are clear: cyan V1, violet V2, green A1, teal A2, amber
  captions.
- Primary actions use dark text on cyan, which passes contrast.
- Icon-only controls have accessible labels/titles; focus-visible styling exists.
- Mobile Setup is readable as a single column; mobile Generate/Edit keep the
  primary action visible.

## Resolution — 2026-11-18

The F7 plan was executed the same day. The findings below remain as the audit
trail; this table is the fix record.

| Finding | Resolution | Evidence |
|---|---|---|
| F7-01 | Shared save vocabulary and tone; load sets `Saved`, edits set `Unsaved`; runtime job fields do not dirty | `stores/saveStatus.ts`, `saveStatus.test.ts`, save pill/footer agree |
| F7-02 | Shot `status`, take state, and progress hydrate from `/projects/{id}`; completed probe shots render `Done` | live probe: 7/7 completed, `6 of 6 shots have takes` |
| F7-03 | `/uploads?project_id=` filters assets and jobs; `missing` flag returned | `test_uploads_project_scope.py`; probe Library shows 1 scoped asset, not 92 |
| F7-04 | Mobile rail 56px, Story spine is a drawer, board 334px at 390 | `ux-matrix.spec.ts` mobile board ≥320 |
| F7-05 | Memory notes render as wrapping `st-warning-note` rows with title; chip max-width no longer applies | `MemoryPanel.tsx`, `.st-warning-note` |
| F7-06 | Generate params rail hides when Inspector opens; per-surface rail defaults/memory | live: stage 784 default, 744 with Inspector, params hidden |
| F7-07 | `MediaThumb` renders an explicit “Media unavailable” placeholder | Vitest/E2E media paths |
| F7-08 | Inspector defaults closed on Library/Deliver/Setup/Shorts and remembers per surface | `DEFAULT_INSPECTOR_BY_SURFACE` |
| F7-09 | Five tracks fit 900px; `Tracks 5/5` visibility menu; visible scroll on short viewports | live: scrollHeight === clientHeight 299 |
| F7-10 | Title form grouped; add action on its own row; color swatch instead of raw bar | Deliver screenshot/JSX |
| F7-11 | Titles preview in the Edit stage; LUT canvas attempted for `.cube` with an export-only badge | **Corrected 2026-11-18:** canvas exists but WebGL upload is blocked by cross-origin taint. See [[f7-caveats-audit-2026-11-18]] F8-01 |
| F7-12 | Command dock is collapsible; mobile status strip keeps device/queue/save only | `CommandDock.tsx`, mobile CSS |
| F7-13 | Asset/take selection, bulk delete, context actions, middle ellipsis | Library surface |
| F7-14 | ComfyUI panel split into Run job / Bindings / Outputs tabs | `ComfyUIPanel.tsx` |
| F7-15 | Story quick actions raised to 36px (≈31px at 0.85 camera) | live control-floor check |
| F7-16/17/19 | Neutral zero states, pluralization, Pending/Running/Done/Failed vocabulary | `shotStatusLabel`, copy pass |
| F7-18 | Middle ellipsis with full-name tooltip | `middleEllipsis` in Library |
| F7-20 | Warning notes wrap and expose the full text via title | CSS + MemoryPanel |
| F7-21 | Status strip prioritizes device/queue/save on mobile | responsive CSS |
| F7-22 | Color/range inputs styled to the dark tokens | `.st-color-field`, accent-color |
| F7-23 | Empty/missing/error media states share one placeholder | `MediaThumb` |
| F7-24 | Pressure tone turns warning when notes or swap ≥90% exist | MemoryPanel |

Verification: 225 backend tests, 102 Vitest, 27 Playwright, axe clean on all
eight surfaces at 1440 and 390, ruff/tsc/ESLint clean.

Follow-up: the caveat audit [[f7-caveats-audit-2026-11-18]] re-tested the LUT,
relink, save-state, contract, and mobile paths and filed `F8-01…F8-18` with the
resolution plan [[../roadmap/f7-caveats-plan-2026-11-18]].

## P1 findings

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| **F7-01** | **Top bar says “Unsaved” in green after a successful load** | `loadProject` sets `saveStatus: 'idle'` (`projectSlice.ts:353-358`); `TopBar` maps `idle → “Unsaved”` and `idle → tone good` (`TopBar.tsx:53-58`) | Users cannot tell whether the project is saved; the status strip simultaneously says `Save idle` |
| **F7-02** | **Shot status is not hydrated from the backend** | Mapping in `projectSlice.ts:266-326` omits `status`, `isGenerating`, `progress`, `stage`, `statusMessage`, `etaSeconds`; completed probe shots render `pending`, and Generate says `0/6 with takes` | Story/Generate/Edit/Review/Library misreport every shot after reload; takes/status work only live via SSE |
| **F7-03** | **Library is global, not project-scoped** | `/uploads` selects all `Asset` rows (`assets.py:257-270`); `useStudioAssets` calls `/uploads` with no project (`useStudioData.ts:351-365`) | 92 media cards (assets + completed jobs) from seven projects appeared in every project; stale cards are confusing |
| **F7-04** | **Mobile Story canvas is only 124px wide** | At 390px: rail 76 + spine 190 = 266; `.st-board-wrap` measured 124px | The board cannot be used on a phone; cards are clipped behind the spine |
| **F7-05** | **Safety/memory warnings are truncated** | Memory notes use `.st-chip` with `max-width:180px` and no title (`studio.css:1285`; `MemoryPanel.tsx:118-125`); live `swap is nearly exhausted (1.4 Gi…` was clipped | The user cannot read the warning that explains a refusal/OOM risk |
| **F7-06** | **Desktop Generate spends 680px on two right rails** | At 1440: stage **424px**, shot params **320px**, Inspector **360px**; prompt/seed/frames duplicate in stage, params, and Inspector | The generation canvas is cramped; common controls wrap; the Inspector competes with the same fields |

## P2 findings

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| **F7-07** | Broken images render as browser-broken icons | `MediaThumb` sets `failed` but the image branch has no fallback (`MediaThumb.tsx:62`) | Library/history cards look broken when files move or old assets remain |
| **F7-08** | Inspector is always present on every surface | Setup/Library/Deliver/Shorts each lose 360px to a shot panel | Less room for the task on hand; Deliver/Library feel narrow |
| **F7-09** | Timeline exposes only 3 of 5 tracks at 900px | Timeline scroll `clientHeight=249`, `scrollHeight=441`; A2/C1 sit below the viewport; command dock starts at y=823 | A2/captions are easy to miss; no visible “more tracks” affordance |
| **F7-10** | Deliver title form hierarchy is awkward | “Add title” sits between fields; native color input renders as a full white bar (`DeliverSurface.tsx:458-495`) | The form reads as unfinished; color choice is visually loud |
| **F7-11** | Titles/LUT are not previewed in Edit | Captions render in the stage; title overlays and LUT only appear in the final render | The edit preview disagrees with the exported frame |
| **F7-12** | Command dock and toasts overlap the timeline/footer | Mobile: dock 767–820, status strip 820–844; Story toast sits above the dock | Bottom content is obscured during generation/feedback |
| **F7-13** | Library asset cards lack media fallback/error state and bulk actions | Delete is a red button on every card; no selection or bulk delete | Noisy cards, tedious cleanup, no recovery for missing media |
| **F7-14** | New ComfyUI panel is dense in a 360px rail | Workflow select, 13 actions, textarea, bindings, node inputs, batch, output cards all live in one inspector | High cognitive load; console-like density returns |
| **F7-15** | Story card quick actions are 26×26 | Four hover actions measured 26×26, below the 28px control floor | Small touch targets and a regression from the F6 28px floor |

## P3 findings — language and visual semantics

| ID | Finding | Example |
|---|---|---|
| F7-16 | Misleading success tone for zero/neutral states | “Unsaved” green; “0 approved” green |
| F7-17 | Pluralization/copy is mechanical | “1 candidate windows”; “0/6 with takes”; “embedded audio mixed” |
| F7-18 | Filenames overflow/wrap aggressively in cards | Long Comfy/job filenames fill 3–4 lines without a middle ellipsis or tooltip |
| F7-19 | Status vocabulary is inconsistent | `pending`, `queued`, `Save idle`, `Unsaved`, `0/6 with takes` |
| F7-20 | Warning chips have no `title`/expand affordance | Memory notes and “Lock seed · deterministic re-run” clip at 178/180px |
| F7-21 | Status strip is fixed at 24px/11px | On mobile the items sit at the edge; no prioritization/overflow |
| F7-22 | Native controls break the dark visual system | Color input is a white browser bar |
| F7-23 | Empty/error media states are inconsistent | Broken `<img>` icon vs. styled empty stage |
| F7-24 | “Normal” memory state is green while the same panel shows swap exhaustion | Tone conflicts with the warning copy |

## Missing functionality

1. **Project-scoped Library** and bulk delete; hide/repair stale assets.
2. **Status hydration after reload** for shots and takes.
3. **Mobile shell mode:** spine as a drawer/overlay, board full width, rail
   collapsible; current 124px canvas is not viable.
4. **Media error placeholders** and a “repair missing media” action in Library.
5. **Edit preview parity:** titles, LUT, and speed/keyframes should match the
   rendered frame.
6. **Inspector policy per surface:** auto-collapse/remember on Library, Deliver,
   Setup, Shorts; de-duplicate Generate’s params and Inspector.
7. **Timeline track affordance:** visible vertical scroll or a “Tracks” control;
   keep A2/C1 reachable above the dock.
8. **Bulk operations** for assets/takes and a context menu instead of 222 visible
   buttons on the Library surface.

## Recommended design direction

- **One primary right rail per surface.** Generate: Inspector or shot params, not
  both. Deliver/Library/Setup: Inspector collapsed by default unless a shot is
  selected and the surface needs it.
- **Truth before tone.** Save state should be `Saved`/`Saving…`/`Unsaved` with a
  neutral or warning tone for unsaved, never success green.
- **Status is derived state, not a default.** Hydrate `status`, progress, stage,
  and take state from the project/jobs on load; do not show `pending` for a shot
  with a completed video.
- **Project-scoped media.** Every Library query should carry `project_id`; add a
  clear orphan/stale asset filter and a safe repair path.
- **Mobile is a different shell, not a narrower desktop.** Full-width board,
  spine drawer, bottom navigation for surfaces, collapsible command dock.
- **Warnings must be readable.** Let memory notes wrap or open a detail popover;
  never ellipsize a safety warning without a title/expand affordance.

## Implementation plan

Phased F7 plan and acceptance criteria:
[[../roadmap/frontend-ux-plan-2026-11-18.md]].
