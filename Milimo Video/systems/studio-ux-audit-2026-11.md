---
type: systems
created: 2026-11-17
updated: 2026-11-17
tags: ux audit studio frontend layout responsive density accessibility
sources:
  - 2026-11-17 Studio screenshot (2048×1285)
  - web-app/src/studio/
  - web-app/src/studio/studio.css
related:
  - studio-shell.md
  - architecture.md
  - frontend-state.md
  - ../roadmap/frontend-refactor-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
---

# Studio UX Audit — 2026-11-17

## Verdict

The Studio is functionally complete, but its chrome is still **desktop-dense and
fragile below ~1200px**. The screenshot exposes four concrete defects, not just
taste:

1. the mode rail's labels are clipped at every viewport;
2. the Inspector has eight tabs in 319px with a hidden scrollbar;
3. the board toolbar overflows its container from 1440px down;
4. the top-bar surface nav collides with the project switcher around 1024px.

The result is a shell that reads as "crammed" even on a 2048px display: 7–10px
labels, 26px status pills, 27px buttons, and multiple horizontal scroll regions
with no scroll affordance. This is a chrome/layout problem, not a missing-surface
problem.

## Method

- Reviewed the provided 2048×1285 Studio screenshot.
- Read the Studio components and `studio.css`.
- Measured live layout with Playwright at **2048, 1440, 1280, 1024, 900, 768**.
- Checked component `scrollWidth` vs `clientWidth`, button heights, label clipping,
  and `elementFromPoint` hit targets.

## Measured evidence

| Viewport | Mode rail | Inspector tabs | Board toolbar | Result |
|---|---|---|---|---|
| 2048 | rail 58px; labels 53–85px wide inside 40px buttons | 319px viewport / **486px content**, hidden scrollbar | 1394 / 1394 | Rail labels clipped; inspector tabs clipped |
| 1440 | same | 319 / 486 | **786 / 1167** | 381px of toolbar outside the container |
| 1280 | same | 319 / 486 | **626 / 1167** | 541px overflow; `Add shot` onward clipped/under inspector |
| 1024 | same | 319 / 486 | **730 / 1167** | 437px overflow; top-bar nav + project name **overlap** |
| 900 | same | hidden | 652 / 1167 | Surface tabs hidden; rail is the only nav |
| 768 | same | hidden | 520 / 1167 | Rail-only navigation with clipped labels |

At 1024px, hit-testing the top-bar tabs returns `.st-project-name` for
**Generate** and **Edit**, and `.st-icon-button` for **Review**. Those surfaces
cannot be clicked where they visually appear.

Density counts from the live surfaces:

| Surface | Buttons | Buttons <28px high | Text nodes <10px |
|---|---:|---:|---:|
| Story | 22 | 15 | 5 |
| Generate | 9 | 3 | 22 |
| Edit | 33 | 27 | 21 |
| Deliver | 10 | 9 | 23 |
| Setup | 22 | 22 | 21 |

## Severity findings

### P0 — Top-bar navigation collides and disappears

- `.st-topbar` is a single 52px flex row with brand, centered surface tabs, and a
  356px right cluster.
- At 1024px the brand shrinks to 111px but the project input still extends to
  x=369 while surface tabs begin at x=141. The tabs overlap the project switcher.
- The only breakpoint that hides surface tabs is `max-width: 900px`
  (`studio.css` near line 2377). The broken 901–1100px band has no fallback.
- At ≤900px the only nav is the 40px mode rail, which is itself clipped.

**Required:** move surface navigation into the rail at <1280, or add a compact
surface dropdown; never allow nav items to overlap project/status controls.

### P0 — Board toolbar is a horizontal junk drawer

`.st-board-toolbar` is a 38px row containing 15+ controls:

- view switch (Board / Shot list / Script)
- tools (Select / Area / Link + link-kind select)
- search
- Add shot / Note / Refs / Arrange
- conditional Align / Dist H / Dist V
- Fit / Minimap / Follow / Pin
- Collab / Undo / Redo / Reset

It has no wrap, no overflow menu, and no scroll container. At 1440 the content is
**1167px** inside **786px**; controls are simply painted under the Inspector.

**Required:** group into primary view/tools and a "More" menu; make search flexible;
hide secondary tools behind an overflow at <1600; use a two-row header or a
dedicated toolbar component.

### P0 — Inspector tabs hide the active section

`.st-inspector-tabs` has `overflow-x: auto` plus `scrollbar-width: none`. The eight
tabs are **486px** wide in a **319px** area. When Effects/Review/History is active,
the first tabs scroll out; the screenshot shows the leading label cut to `tion`.

**Required:** segmented icon+label tabs that fit, a dropdown for secondary tabs, or
visible previous/next scroll buttons. The active tab must remain pinned/visible.

### P0 — Mode rail labels are all clipped

`.st-rail` is 58px; `.st-rail-button` is 40px with 7px uppercase labels. Measured
label widths are 53–85px, so every label overflows its button. `Studio`, `Legacy`,
and `Setup` also overflow the 57px rail itself.

**Required:** either an icon-only rail with tooltips and an accessible label, or a
72–80px rail with 10px labels and no clipping. The label must not bleed into the
Story spine.

### P1 — Typography is below comfortable scanning size

- Rail labels: 7px.
- Inspector tabs: 10px.
- Most status pills/metadata/chips: 9–10px.
- Buttons: 10–11px; small buttons 10px.
- Generate/Deliver/Setup each have 21–23 text nodes under 10px.

This is the main reason the UI feels "swished": hierarchy is carried by tiny
all-caps labels rather than size, spacing, and weight.

**Required:** 12–13px body/control base, 11px metadata, 10px only for true group
labels; reserve all-caps for section labels, not every button.

### P1 — Controls are below a reliable target size

- 27 Edit buttons are under 28px high.
- All 22 Setup buttons are under 28px.
- Status pills are 26px high.
- Inspector tabs are 26–28px high.

**Required:** 32px minimum for normal controls, 28px only for dense table actions;
40px rail buttons. This also improves touch/tablet use.

### P1 — Competing status and primary actions

The top bar carries Save, GPU free, needs-you, undo, redo, inspector toggle, share,
and backend status at equal visual weight. The board toolbar mixes creation,
selection, layout, collaboration, and history. There is no contextual primary
action ("Generate selected", "Export", "Review next").

**Required:** one context-aware primary action per surface; move status into the
bottom status strip or an overflow popover; keep save + health summary.

### P1 — No quick actions on cards or clips

`ShotCard` is one large `<button>`; its only actions are select, drag, and resize.
Generate, open in Edit, duplicate, delete, pin, and review are in the Inspector or
elsewhere. The board's empty shot card has a large blank media region and a tiny
status badge.

**Required:** hover/focus action row on cards; right-click context menu; empty-state
CTA inside the card ("Generate preview", "Open in Edit").

### P1 — Recall and discoverability

- No command palette with searchable commands; the dock is an agent prompt.
- No shortcut cheat sheet.
- No "More" menus, so tools are either visible or lost.
- Icon-only controls rely on `title`; several have no explicit `aria-label`.
- Hidden horizontal scrolls in the Inspector and command context row have no
  visible affordance.

### P2 — Fixed panes and no density/layout preferences

- Story spine is fixed at 236px, 190px below 900px; it has no collapse or resize.
- Generate is a fixed 220 / 1fr / 280 grid; at 1024 the stage is squeezed.
- Inspector is 360px by default; its tabs need more room than the panel.
- No compact/comfortable density mode, no saved layout presets.

### P2 — Command dock is always present when open

The command dock consumes a full-width strip with four modes, a prompt, Plan, and a
hint. It is useful but visually heavy on every surface.

**Required:** collapsed "Ask/Plan" pill by default; expand on `⌘K`; remember the
open state per surface.

### P2 — Minor card and table overflows

Library take cards show 2–20px internal horizontal overflow at 1024. These are
small, but they reinforce the "things are squeezed" impression.

## Missing functionality

1. Responsive top-bar navigation and a compact surface switcher.
2. Toolbar overflow menus and grouping.
3. Inspector tab overflow/menu with active-tab visibility.
4. Icon-only rail mode with tooltips, or a wider labelled rail.
5. Bulk action bar for multi-selected cards (duplicate, delete, pin, tag, export).
6. Card/timeline context menus.
7. Global command palette and shortcut help.
8. Collapsible/resizable story spine, Generate panes, and Inspector.
9. Density/layout preferences persisted per user/project.
10. Contextual primary action in the top bar.
11. Visible scroll affordances for any remaining horizontal scroller.
12. Per-surface visual-regression baselines at 2048/1440/1024/900/768.

## Improvement direction

### Layout and spacing

- Add spacing tokens: 4 / 8 / 12 / 16 / 24 / 32.
- Use 12px as the control/body baseline, 11px metadata, 10px section labels.
- Give toolbars 48–56px height and group controls with separators.
- Reserve one row for primary controls; move secondary actions into a `⋯` menu.
- Use 32px minimum control height and 40px rail buttons.

### Top bar

- `≥1280`: brand + project + centered surface tabs + save/health/share.
- `1024–1279`: brand + project + compact surface dropdown + save/health.
- `<1024`: rail/compact nav; status in the bottom strip.
- Never allow tabs to overlap project/status controls.

### Mode rail

- Preferred: 72px labelled rail with 10px labels.
- Alternative: 56px icon-only rail with persistent tooltips and an accessible name.
- Active state remains a filled accent tile; no text clipping.

### Inspector

- Keep 3–4 primary tabs visible (Shot / Takes / Audio / Effects).
- Put Motion / Review / History into a `More` menu or icon strip.
- Pin the active tab; add scroll chevrons if horizontal scrolling remains.
- Use collapsible sections with a sticky section header.

### Board toolbar

- Row 1: view switch, search, Add shot, primary layout action.
- Row 2 (or `More`): tools, refs, arrange, fit, minimap, follow, pin, collab,
  undo/redo/reset.
- Search width flexes; at <1280 it collapses to an icon that expands.
- Group with separators; do not mix creation, navigation, and destructive actions
  without hierarchy.

### Cards and selection

- Add a hover/focus action row: Generate, Open in Edit, Duplicate, More.
- Add a contextual bulk bar when 2+ cards are selected.
- Show empty-state CTA directly in the card media area.
- Increase card meta from 9px to 11px; give the title two lines.

### Command palette

- Separate "run a command" from "ask the agent".
- `⌘K` opens a searchable palette: surfaces, commands, projects, shots, references.
- Keep the agent dock as `⌘J` or an expandable panel.

## Proposed implementation plan — F6 Chrome stabilization

Add these as the next front-end phase.

| Task | Priority | Change | Acceptance | Effort |
|---|---|---|---|---|
| F6-01 Responsive top bar | P0 | Compact surface switcher at <1280; no overlap | All surfaces clickable at 1024/1280/1440 | S |
| F6-02 Board toolbar groups + overflow | P0 | Primary row + `More` menu; flexible search | No child outside toolbar at 1440/1024; all actions reachable | M |
| F6-03 Inspector tab overflow | P0 | Primary tabs + More menu/chevrons; active pinned | Active tab always visible; no hidden scrollbar | S |
| F6-04 Mode rail width/labels | P0 | 72px labelled or icon-only + tooltip | No clipped label at 2048–768; label in a11y name | S |
| F6-05 Type and spacing tokens | P1 | 12/13px base, 11px meta, spacing scale | No body text <11px; section labels only at 10px | M |
| F6-06 Control heights | P1 | 32px min; 40px rail; table 28px exception | All primary controls >=32px | M |
| F6-07 Status hierarchy | P1 | Move secondary status to status strip/overflow | One context primary action; no equal-weight noise | S |
| F6-08 Card quick actions + bulk bar | P1 | Hover actions, context menu, multi-select bar | Generate/open/duplicate/delete without Inspector | M |
| F6-09 Collapsible panes | P2 | Spine, Generate panes, Inspector, command dock | Persisted collapse state; stage remains usable | M |
| F6-10 Command palette + shortcuts | P2 | Searchable commands, shortcut sheet | `⌘K` reaches every surface/action; help panel | M |
| F6-11 Density/layout preference | P3 | Comfortable/compact + saved layout | Preference survives reload | S |
| F6-12 Responsive visual tests | P0 | 5 widths × key surfaces; overflow assertions | CI fails on nav overlap/toolbar overflow/label clipping | S |

**F6 gate:** no overlapping top-bar controls, no clipped rail/tabs/toolbars, no
hidden active inspector tab, no body text below 11px, primary controls ≥32px, all
surfaces reachable at 2048/1440/1024/900/768, and axe/keyboard gates still green.

## What not to change

- Do not add more controls to the board toolbar until the grouping/overflow work is
  done.
- Do not shrink type to fit more; move secondary controls into menus.
- Do not replace the board/timeline engines for this pass. This is chrome, density,
  and responsive behavior.

## Related

- [[studio-shell.md]] — shell architecture and surface map
- [[frontend-state.md]] — layout/selection state
- [[../roadmap/frontend-refactor-2026-11.md]] — F6 plan
- [[../roadmap/implementation-plan-2026-11.md]] — remaining/deferred items
