---
type: systems
created: 2026-11-17
updated: 2026-11-17
tags: media preview library story generate edit review audit
sources:
  - 2026-11-17 Studio media report
  - web-app/src/studio/
  - backend/routes/assets.py
related:
  - architecture.md
  - studio-shell.md
  - studio-ux-audit-2026-11.md
  - ../roadmap/implementation-plan-2026-11.md
---

# Studio Media Audit — 2026-11-17

## Symptom

In the new Studio, images and videos rendered but could not be clicked or opened
from Library cards, board cards, take grids, references, or review rows. Only the
Generate stage and Review stage exposed usable media, and even there the image had
no full-size viewer.

## Root causes

1. **No global media viewer.** Every surface rendered `<img>`/`<video>` as passive
   content. Only a few surfaces had native `controls`.
2. **Cards had no preview path.** Story cards were selection/drag surfaces; Library
   assets, takes, references, Inspector takes, and Review takes had no click action.
3. **The board card select layer captured clicks.** The accessible overlay selected
   the shot, but there was no single-click preview action.
4. **Legacy absolute media URLs bypassed the current API.** `getAssetUrl()` returned
   `http://localhost:8000/...` unchanged, so LAN/Tauri clients and bearer-token
   sessions could load project data but not media.
5. **Captions were classified as video.** The upload route mapped `.srt`, `.vtt`,
   `.json`, `.txt`, and `.md` to `video`, so a caption file opened in a `<video>`.
6. **Hidden card actions still received clicks.** The hover action row used
   `opacity: 0` without disabling pointer events.
7. **Videos had no poster frames.** Uploaded videos had no `thumbnail_url`; even
   generated videos exposed posters to the API but the Library rendered a passive
   `<video>` instead of the poster.
8. **No hover playback.** Cards showed a static frame or black media until opened.

## Fix

- Added a global media-preview store and `MediaPreview` dialog with:
  image, video with controls, audio, and text/caption rendering; Escape/backdrop
  close; open-original and download links.
- Wired preview into:
  - Library assets, takes, references, and search results
  - Story board cards (hover action, double-click, context menu)
  - Master-reference board cards (double-click)
  - Generate selected image, take grid, and preview still
  - Inspector take grid
  - Review take rows and main review image
  - Edit transport and double-click on the stage
- Rewrote legacy localhost media URLs onto the current API origin and kept bearer
  tokens on asset URLs.
- Classified uploads as `image`, `video`, `audio`, `caption`, or `text`; the Library
  infers the kind from the filename as a fallback for old rows.
- Disabled pointer events on hidden card action rows.
- Added `Asset.thumbnail_url` (Alembic `0004_asset_thumbnail`) and generated real
  poster frames on upload; `/uploads` backfills missing legacy video posters.
- Added `MediaThumb`: poster-first thumbnails, muted looping hover playback, play
  badge, and fallback to the video frame if a poster fails.
- Renders now generate a poster too, shown in Deliver/Shorts history.

## Surface audit

| Surface | Before | After |
|---|---|---|
| Library assets | Passive image/video | Click opens image/video/audio/text preview |
| Library takes | Thumbnail only | Click opens the take media |
| Library references | Passive image | Click opens reference image |
| Library search | Assets/references not openable | Opens the matching media |
| Story board | Select/drag only | Hover Preview + double-click + context menu |
| Board references | Drag only | Double-click preview |
| Generate | Current video controls; images passive | Current image, preview still, and take thumbnails open |
| Inspector takes | Passive thumbnail | Click preview |
| Edit | Stage only | **Preview** button + double-click stage |
| Review | Video controls; image passive | Main image opens; take rows have Preview |
| Shared review | Video controls | Full-frame media remains read-only |
| Video cards | Black/passive video | Poster + muted hover playback |

## Verification

- Manual: Library rendered **83** media-open buttons; image and video previews
  opened; video reported `readyState=4` and played.
- E2E: `media-preview.spec.ts` uploads a real image, opens it from Library, verifies
  the dialog/image, and closes with Escape.
- Final suite: **84 pytest, 56 Vitest, 17 Playwright**, axe green, tsc/ESLint clean,
  build + bundle pass.

## Remaining polish

- The viewer is single-item; there is no next/previous gallery navigation.
- Images have no zoom/pan controls in the viewer.
- Text/caption preview is plain monospace, not an editable caption editor.
- Shared review keeps a read-only stage rather than opening a second viewer.
