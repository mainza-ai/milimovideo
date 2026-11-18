# Milimo Video — User Guide

Milimo Video is a local-first AI film studio: a non-linear editor, storyboard,
generation workspace, review room, and delivery desk in one application. It runs
on your machine. No cloud account is required for the cut, the project data, or
the render.

There are two interfaces:

- **Studio** — `http://localhost:5173/studio`. The current shell described here.
  The scoped P0–P5 / F0–F5 implementation plan is complete.
- **Legacy editor** — `http://localhost:5173/`. Frozen fallback for the cutover
  window; set `VITE_STUDIO_ENABLED=false` to serve it at `/studio` as a rollback.

The desktop shell (Tauri) opens the same Studio shell in a native window.

---

## 1. Install and launch

### Hardware

- **Apple Silicon:** 32 GB unified memory or more recommended for the 19B video
  model. MPS is supported; long jobs are chunked.
- **NVIDIA:** 16 GB+ VRAM recommended.
- **Disk:** model checkpoints are large. Setup reports free disk before download.
- **FFmpeg/ffprobe** must be installed and on `PATH` for render and media probing.

### First-time setup

```bash
python3 -m venv milimov
./milimov/bin/pip install -e ./LTX-2/packages/ltx-core \
  -e ./LTX-2/packages/ltx-pipelines -e ./flux2 \
  -r backend/requirements.txt
cd web-app && npm install && cd ..
```

### Launch

```bash
./run_backend.sh     # backend on http://127.0.0.1:8000
./run_frontend.sh    # Vite on http://127.0.0.1:5173
```

Optional:

```bash
./run_sam.sh         # SAM 3 segmentation on port 8002
```

`run_backend.sh` runs `verify_setup.py` first and refuses to start with a missing
import, FFmpeg, model path, or unwritable directory. Fix the reported item instead
of bypassing the check.

### Desktop shell (Tauri)

```bash
cd web-app
npm run tauri dev
npm run tauri build
```

The desktop shell uses the same Studio UI and provides native Open and Save
dialogs for storyboard backups and pipeline recipes. If the backend is offline,
the top-bar status offers **Start backend**, which runs `run_backend.sh` from the
checkout. Bundled Python sidecar packaging is still a follow-up.

### Health

```bash
curl http://127.0.0.1:8000/health
```

The response reports device, GPU busy/queued, active memory slots, auth mode, and
free disk. Every HTTP response carries `X-Request-ID`; backend logs include it.

---

## 2. Orientation

The mode rail on the left switches surfaces. The top bar holds project switching,
save status, undo/redo, review count, health, and Share.

| Surface | Purpose |
|---|---|
| **Story** | Script, story spine, board, shot list, notes, references, links |
| **Generate** | Shot prompt, ingredients, takes, previews, batch queue, inpaint |
| **Edit** | Stage, compare, timeline, tracking, blocking, 3D previz |
| **Review** | Frame comments, approvals, activity, compare |
| **Library** | Assets, master references, takes, ranked search |
| **Deliver** | Presets, range, captions, render history |
| **Shorts** | Transcript candidates and vertical/square exports |
| **Setup** | Device, queue, pipelines, models/LoRAs, error budget |

Useful shortcuts:

- `1`–`8` switch surfaces.
- `Space` play/pause (opens Edit when needed).
- `⌘K` / `Ctrl+K` opens the command palette.
- `⌘J` / `Ctrl+J` toggles the agent command dock.
- `?` opens the keyboard shortcut sheet.
- The palette can switch density (comfortable/compact), toggle panes, and reach
  every surface.
- `⌘Z` / `Ctrl+Z` undo; `⇧⌘Z` / `Ctrl+Shift+Z` redo.
- `B` blade at the playhead, `R` ripple-delete, `[` / `]` slide the selected V1 clip.
- `Escape` exits focus mode.
- `Shift`-click cards to multi-select; `Shift`-drag empty canvas to marquee.

---

## 3. Projects and saving

- The project switcher is next to the project name. It lists server projects,
  creates new ones, and deletes the current project with confirmation.
- Renaming the project saves on blur; the Save control flushes any pending edits.
- If the project remembered by the browser no longer exists, Studio opens the most
  recently updated server project instead of editing a phantom document.
- Saves are serialized. The top bar shows **Unsaved / Saving / Saved / Save failed /
  Conflict**.
- A revision conflict means another client wrote first. Reload before saving; do not
  overwrite blindly.
- Undo is document-scoped. Board moves, links, areas, and notes use a separate board
  undo, so layout experiments do not consume document history.

Project data includes shots, scenes, caption cues, script text, board layout/view,
render preset, audio settings, keyframes, blocking nodes, and review/share records.

---

## 4. Story

### Script

Paste a screenplay or numbered shot list in **Story → Script**.

- **Save script** persists the raw text.
- **Parse & replace storyboard** runs the local parser, archives the current
  storyboard first, and replaces scenes/shots after confirmation.
- **AI parse (queued)** dispatches the heavy parse through the GPU queue instead of
  blocking the request. It polls and commits when complete.
- **Download backup** downloads the pre-parse archive as JSON.

### Story spine

The left spine groups scenes and shots. Search filters the spine; master-reference
chips filter to shots carrying that reference.

### Board

- Pan: drag empty canvas.
- Zoom: `⌘`/`Ctrl` + wheel, or the zoom pill.
- Select: click a card; `Shift`-click to add/remove.
- Marquee: `Shift`-drag empty canvas.
- Move: drag a selected card; all selected cards move together.
- Resize: drag the bottom-right corner.
- Arrange/Fit: toolbar buttons.
- Align/Distribute: appears with 2+ selected cards (distribute needs 3+).
- **Area**: drag on empty canvas to create; rename inline, change/collapse it, drag
  its header to carry contained cards.
- **Link**: choose Link mode, click two cards, pick thread/blocking/reference.
- **Note**: adds an editable board note; drag by its header.
- **Refs**: shows draggable master-reference cards.
- **Pin**: keeps the selected card out of auto-placement.
- **Collab**: opt-in Yjs board sync between tabs on the same machine. Positions and
  pins converge over BroadcastChannel; no server is involved.

Review comments and change requests appear as task cards beside their shots.

### Shot list and script views

The list view is the accessible equivalent of the board: multi-select, bulk delete,
inline prompt/shot-type editing, and a typed-link list. Script view is editable and
persists.

---

## 5. Generate

Select a shot in Generate to work on its prompt, ingredients, parameters, and takes.

### Ingredients

- Click a master reference chip to attach/detach it.
- **Insert @trigger** appends the trigger word to the prompt.
- Reference images are passed to image/video generation when available.

### Advanced parameters

Open **Advanced parameters** for negative prompt, seed, frame count, width/height,
CFG, pipeline override, prompt enhancement, and upscale.

Keep LTX frame counts at `8k+1` (for example 49, 81, 121). The pipeline lens warns
when the count is invalid. The lens copies/saves a reproducible recipe as JSON and
**Import JSON** loads one or a `{name, recipes}` pack into the draft before you
apply it.

### Generate, cancel, queue

- **Generate** queues one shot through the durable GPU queue.
- **Seed lock** is on by default: the same prompt and parameters reproduce the
  same take. Unlock it in **Advanced parameters** (or the Inspector) to enable
  **New take**, which randomizes the seed while preserving old takes.
- **Cancel** requests cancellation; the queue state machine converges.
- **Queue pending** batches all shots without completed takes.
- **Batch matrix** shows status/progress and per-shot generate/cancel.
- **Active queue** shows running/queued jobs, position, estimate, and cancel.
- **Pipeline** in Advanced parameters selects `auto`/LTX, **wan 2.2 (5B)**, or
  **LTX-2.5** (gated weights). Wan is a separate low-VRAM text-to-video engine;
  LTX-2.5 uses the gated diffusers adapter and native audio. **Steps**,
  **Priority** (lower = earlier), and **Transition** (V1 crossfade seconds)
  persist with the shot.
- Only one heavy GPU job runs at a time; interactive video/image work is dispatched
  before bulk thumbnails/elements and background parsing.
- **Memory preflight** runs before the request. If the shot cannot fit:
  - a running job holds the memory → the shot queues and shows **waiting for
    memory** instead of failing;
  - a smaller preset fits → a banner offers **Apply & generate** with the largest
    safe preset (upscale off, fewer frames, smaller resolution);
  - nothing fits → the shot is blocked with the estimate and suggestions.
- **Progress** appears inline (stage, percent, ETA), as a toast, in the top-bar
  activity center (active/history/retry/cancel), in the document title, and on the
  macOS dock badge. Reconnects replay missed events; a missed terminal is
  reconciled from `/status`.
- Terminal notifications can be muted for the current project in **Setup →
  Notifications**; toasts, desktop notifications, and sound are separate toggles.

### LTX-2.5 (gated)

The `ltx25` pipeline runs the official `Lightricks/LTX-2.5-Diffusers` directory
with native audio, CPU offload, and VAE tiling. Weights are gated: accept the
community license on Hugging Face, download the diffusers directory in Setup (or
set `MILIMO_LTX25_MODEL`), then choose **LTX-2.5** in Advanced → Pipeline. It is a
single-shot engine; the chained multi-chunk path is LTX-2 only.

### ComfyUI workflows

ComfyUI is an optional local engine, separate from LTX-2/Flux/Wan. Core
generation never calls it; `MILIMO_COMFY_ENABLED=0` removes the panels and API
surface, and the Generate panel stays hidden while ComfyUI is offline. Milimo
does not edit node graphs; it runs a workflow you export from ComfyUI.

1. In ComfyUI, build the graph and choose **Workflow → Save (API Format)**. The
   UI-saved JSON (`nodes`/`links`) is rejected; the API format is required.
2. In **Generate → ComfyUI workflow**, paste the JSON or **Import JSON**. The last
   workflow is remembered locally. **Smoke test** loads a model-free
   `EmptyImage → SaveImage` graph that proves the bridge.
3. **Validate** reports node count, output nodes, and inferred size.
4. **Run workflow** queues it against the selected shot. Progress streams like any
   other job (queue position, node execution, sampler steps, saving).
5. Every output is copied into the project and appears in Library; the primary
   image/video is attached to the shot. **Cancel** deletes/interrupts the ComfyUI
   prompt.

Reuse a saved graph with different inputs by editing the JSON, or pass per-node
`overrides` through the API. **Setup → ComfyUI bridge** shows reachability,
version, queue depth, free RAM, and **Free cached models** (`/free`).

ComfyUI is admitted as a declared external slot (`MILIMO_COMFY_GB`, default 2 GB)
and competes for the same GPU queue. ComfyUI defaults to port 8001 and SAM 3 to
8002, so both can run. If ComfyUI is elsewhere, set `MILIMO_COMFY_URL` (for
example `http://127.0.0.1:8188`). See [Using the ComfyUI Bridge](comfyui.md) for
the full setup and troubleshooting guide.

### Preview first

**Preview frame** generates a still with the current prompt and attaches it as
frame-0 conditioning. Use **Use as end frame** to attach the same still at the
last frame, or remove either conditioning image from the list. Inspect it before
spending video time.

### Takes and favorites

Completed jobs are immutable takes. Select any take to repoint the shot; the old
outputs remain. Favorite a take to make it the preferred comparison.

### Inpaint / edit frame

In Edit, open **Edit frame**, describe what to replace (SAM text mask) and the new
content, then apply. The current frame is captured, uploaded locally, inpainted,
and selected as a new take. This uses the same GPU queue.

---

## 6. Edit

### Stage and compare

- **Side** and **Wipe** compare the selected take with the previous/favorite take,
  synced to the same clip-local frame.
- Toggle grid, safe-area, and rule-of-thirds guides.
- **Edit frame** opens inpaint.
- **Blocking** opens the top-down diagram.
- **3D** opens the lazy three.js previz.
- **Track** opens object tracking/masks over the live stage.
- **Depth** estimates a local depth map for the current frame and overlays it
  with an opacity control. Download Depth Anything V2 Small in Setup first.
- **Preview** in the transport opens the current clip full size; double-clicking
  the stage does the same.

### Timeline

Tracks:

- **V1** — magnetic story track. Order is `sequence_index`.
- **V2** — free overlay track.
- V1/V2 clips show the shot poster thumbnail behind the title (generated and
  uploaded videos get poster frames).
- **A1 / A2** — audio/music tracks, each with mute/solo.
- **C1** — captions. Titles are project-level overlays, not a track.

Tools:

- **Select**: click clips, drag trim handles at the edges.
- **Slip**: drag a clip to move its source window while keeping duration.
- **Roll**: drag a clip to move the cut with the next V1 clip.
- **Blade**: split at the playhead.
- **Ripple**: delete selected clip and reflow V1.
- **Slide ◀/▶**: move the selected V1 clip earlier/later.
- **Caption**: add a cue at the playhead; edit text/in/out inline.
- **Import**: SRT/VTT captions.
- **Audio**: choose A1 or A2 and place an uploaded audio asset at the playhead.
- **M/S**: mute or solo a track.
- **Table**: editable list of clips, trims, speed, mute, delete.
- Keyboard: `B`, `R`, `[`, `]`; trim handles support arrow keys.

### Rate, trims, fades, keyframes

- Speed is 0.25–4× and changes timeline duration; render applies `setpts`/`atempo`.
- **Transition** (Generate → Advanced) overlaps a V1 clip with the previous one and
  crossfades the incoming clip; the clamp is half of both clips.
- Trim in/out are source frames.
- Audio gain, mute, fade-in, fade-out are per clip.
- Motion keyframes interpolate opacity, scale, pan X/Y, and volume. Add at the
  playhead in Inspector → Motion; each property has its own control.

### Captions

Caption cues are project data. Import SRT/VTT, edit in C1, and choose whether Deliver
burns them into the render. ASS is generated locally and applied after compositing.

### Blocking and 3D

Blocking persists camera/character/object nodes per shot. 3D previz reads those
nodes, draws a camera path across shot order, and supports orbit/zoom/pan plus a PNG
snapshot. WebGL failure is caught; the editor continues without 3D.

---

## 7. Review and sharing

- Pause on a frame and add a comment. Comments store the shot and frame.
- Resolve, reopen, or delete comments.
- Approve a shot or request changes; the latest decision is the shot’s status.
- Activity merges comments, decisions, and generated takes.
- The top bar’s **needs you** pill counts open comments and change requests.
- **Share** copies a capability link. A reviewer opens a read-only page, comments,
  and approves without editor access. Links are revocable and disabled when the API
  token gate is on.
- A comment containing `@Agent` offers **Plan fix in command dock**. The agent
  proposes a diff; nothing is applied until approval. Plan patches are one undo step.

Version stack lets you select a take or compare it against the current one.

---

## 8. Agent command dock

Modes:

- **Ask** — project/shot facts.
- **Direct** — explicit edits (`prompt:`, `negative:`, `seed N`, `frames N`,
  `type close-up`, `add …`). A plan diff is previewed.
- **Generate** — proposes a new take with model/time/VRAM estimate.
- **Find** — filters the board.

Plans show steps, before/after, warnings, and rough local estimates. Approve applies
through the undoable document path; generation only runs after approval.

Recipes save a plan by name and can be reloaded against the current selection. Use
`{{selection}}` in a recipe to insert the selected shot prompt.

---

## 9. Library and search

- **Assets** — upload directly with **Upload media**; images, video, audio, and caption files are supported. Click any media card to open the global preview (video/audio controls, image full size, caption/text view, download). Video posters are generated on upload and the muted video plays while hovering.
- **Master references** — Characters, Locations, Objects, and Styles with trigger
  words. Click a reference image to preview it. **New reference** creates one with a name, type, trigger word, image, and
  description; **Edit** updates it and **Delete** removes it. Trigger words are
  normalized to a single leading `@` and can be mentioned in prompts.
- **Delete assets** — each Asset card has a **Delete** action. It removes the
  file and its generated poster, clears any shot that pointed at it, and marks the
  matching take/job deleted. Confirm the browser prompt to proceed. Master
  references and takes have their own delete actions.
- **Takes** — every completed take with shot lineage, seed/model, select action,
  and click-to-preview media.
- **Search** — local ranked token/prefix search across shots, assets, references,
  and takes. No embedding service; results show the score.

---

## 10. Deliver

- Presets: project, 1080p, 720p, vertical 9:16, square 1:1, portrait 4:5.
- **Render range**: in/out seconds; full-cut reset.
- **Fit** chooses letterbox (**Pad**) or center-crop (**Crop**) for a different output aspect.
- **Encoding** sets CRF (quality/size), encoder speed, audio bitrate, and optional
  loudness normalization. **Add 3D LUT** loads a `.cube`/`.3dl`/`.dat`/`.m3d` file
  and applies it to the V1/V2 composite before captions. These values persist
  with the project.
- **Titles** adds a burned-in text overlay (text, in/out, size, color). Titles
  render after the LUT and before captions.
- **Burn captions** toggles ASS burn-in.
- **Create export** queues a real FFmpeg render through the job ledger.
- Render history shows status, progress, errors, an output link, and **Cancel** for
  queued/running renders.
- **Export OTIO** or **Export FCPXML** writes the computed V1/V2/A1 cut for Resolve,
  Premiere, or Final Cut. **Import OTIO/FCPXML** rebuilds shots from a returned
  timeline, including trims, speed, and project media URLs.
- The preset is stored with the project.

Render truth includes V1/V2 compositing, alpha, embedded audio, A1 mix, trims,
speed, fades, keyframes, captions, and range. The FFmpeg output is the cut.

---

## 11. Shorts

1. Import an SRT/VTT transcript (or fall back to shot text/timing).
2. Pick a reframe preset: 9:16, 1:1, or 4:5.
3. Review scored candidate windows (10–45 s). Score combines word density, hook
   words, and numbers; long silences break windows.
4. Select a candidate and adjust **In/Out** seconds if the suggested window is not
   exactly right; **Reset range** restores the candidate.
5. Export. The render uses the edited range, transcript captions, the chosen
   preset, and center-crop reframing.
6. Recent exports read the same render history.

---

## 12. Setup and models

- Device, active slots, GPU busy/queued, auth mode, free disk, dispatch state.
- **Pause queue** stops new GPU dispatch after the current job; **Resume queue**
  releases waiting durable jobs.
- Pipeline readiness with missing counts.
- Base model table: real status, size, VRAM estimate, Activate/Unload or Download;
  active downloads expose **Cancel download**. **Recommended** entries explain the
  lane: LTX-2.5 for 32 GB+ (gated), LTX-2 as the stable baseline, Wan 2.2 5B for
  low-VRAM video, Flux 2 Q4_K_M for low-VRAM image, plus SAM 3 and depth.
- Quantized Flux is activated with `MILIMO_FLUX_GGUF=/path/to.gguf`; it supports
  text-to-image, not IP-Adapter or inpainting.
- LoRA table: real downloaded/active state, activation, and strength.
- Depth Anything V2 Small is optional; download it here before using the Depth overlay.
- Project integrity: **Scan project** reports missing media, orphan shots/scenes,
  orphan board nodes/links, and invalid caption ranges. **Repair safe issues**
  clears broken pointers, normalizes sequence order, and sanitizes captions
  without deleting files.
- **Memory budget**: measured used/available/ceiling, physical free, free swap,
  pressure, SAM, per-slot model/bytes/leases with **Release**/**Pin**, external LLM
  engines with **Release**, and load/unload/eviction/refusal counters. **Release
  all** frees every resident model; **Pin** skips idle eviction.
- **Notifications**: in-app toasts, sound cue, desktop notifications (macOS uses
  Notification Center), and per-project mute. Toasts/activity always keep history;
  mute only suppresses the interruption.
- **ComfyUI bridge (optional)**: reachability/version, queue depth, free RAM, **Probe**,
  **Free cached models** (`/free`), and a link to the ComfyUI UI. ComfyUI is
  budgeted as a declared external slot (`MILIMO_COMFY_GB`).
- Error budget: 7-day job counts by status and recent failures.

Activation loads the real model through the loader and respects the one-slot rule.
A 32 GB Mac may refuse a model that cannot fit; that is a correct result, not a bug.

### AI & LLM

Studio Setup → **AI & LLM** is the local-first LLM configuration plane. Settings
survive restarts; every write is revision-checked and a stale save is refused with
a conflict instead of overwriting newer state.

- **Providers are instances.** Ollama, oMLX, Strata, LM Studio, vLLM, and
  llama.cpp can run at the same time: each endpoint gets its own card (`ollama`,
  `openai_compatible@127.0.0.1:8787`, …). **Scan local servers** identifies real
  engines by their model catalog, and **Attach** adds or refreshes one. **Add
  endpoint** takes a URL for anything on another port.
- **Capabilities are visible.** **List models** shows context window, parameter
  size, vision, and whether an entry is chat-capable. Embedding and tool/audio
  models are flagged and cannot be assigned to a text role. **Set default** picks
  the model for that endpoint.
- **Roles** (`prompt_video`, `prompt_image`, `script_parse`, `agent_plan`,
  `review_critic`, `continuity`) route independently, support ordered fallbacks,
  and can be left **unassigned**. Gemma only serves `prompt_video` inside a loaded
  LTX pipeline; image and script roles need a local HTTP model.
- **Warm**, **timeout**, and **keep-alive** are per provider. A cold 27B load can
  take ~36 s; warming separates that from inference. `keep_alive: 0` releases the
  model immediately after a call.
- **Check plan** reads live catalogs and names any role model that no longer
  exists on its engine.
- **Run eval** executes shot, image, and script fixtures through the saved plan
  and records pass rate and p50 per role. **Apply recommended** assigns the tested
  models. The evidence is stamped with the settings revision, so a later change
  marks it stale.
- **Budget & profiles.** Cloud roles need `MILIMO_LLM_CLOUD=1`, air-gap off, a
  positive daily budget, and a known per-model rate; unknown prices fail closed.
  With **eval gate** on, a cloud assignment also needs a passing eval for that role.
  **Sovereign local** pulls every role back to a local engine and zeroes the
  budget; **Local preferred** keeps cloud configured but unrouted; **Hybrid**
  allows cloud agent/review fallbacks only when a key, budget, and eval exist.
- **Test saved configuration** streams the reply when the engine supports SSE and
  falls back to one response when it does not (oMLX has done both). It reports the
  provider, model, and latency.
- **Image input** is sent on local HTTP wires (Ollama images, OpenAI image_url,
  Anthropic blocks), so a vision model can analyse a reference frame.
- The legacy header gear remains a two-provider quick switch (`prompt_video` only)
  and links here for everything else.

---

## 13. Data, privacy, and security

- Binding defaults to loopback (`127.0.0.1`).
- CORS is an explicit allowlist.
- `MILIMO_API_TOKEN` enables a bearer-token gate. Share links are disabled in this
  mode because a reviewer would not have the studio token for media.
- Uploads are extension/size checked and streamed; filenames are sanitized.
- Media paths are confined and served with Range support.
- Debug artifacts are off unless `MILIMO_DEBUG` is set.
- No project data leaves the machine by default. LLM cloud providers additionally
  require `MILIMO_LLM_CLOUD=1`, air-gap off, a positive daily budget, and a known
  price; a configured cloud role refuses otherwise.

---

## 14. Verification and troubleshooting

```bash
./milimov/bin/python -m compileall -q backend
./milimov/bin/python -m pytest
./milimov/bin/python -m ruff check backend
./milimov/bin/python backend/verify_setup.py

cd web-app
npx tsc -b
npm run lint
npm test
npm run build
npm run check:bundle
npm run test:e2e
curl http://127.0.0.1:8000/health
```

Common issues:

- **Backend refuses to start:** run `verify_setup.py`; fix the exact missing import,
  FFmpeg binary, model path, or directory permission.
- **GPU busy forever:** check `/health` queue/slots; inspect `/metrics/jobs`; cancel
  the job from Generate or Setup.
- **OOM:** the scheduler retries once at 75% resolution and an `8k+1` frame count.
  If it fails again, reduce resolution/frames or unload another model.
- **Save conflict:** reload the project before saving; another client wrote first.
- **Black render:** check the shot has media, V1 order, tracks hidden, and trims.
- **No captions:** verify cues exist, caption burn is on, and the render range
  intersects the cue timings.
- **3D blank:** WebGL may be unavailable; the modal reports it and the editor stays
  usable.
- **A role stopped working:** Setup → AI & LLM → **Check plan** lists role models
  missing from a live catalog. Reassign the role or attach/verify the engine.
- **Provider unreachable:** use **Verify**; check the base URL, that the engine is
  running, and `needs key` for authenticated endpoints.
- **Cloud call refused:** cloud needs `MILIMO_LLM_CLOUD=1`, air-gap off, a positive
  daily budget, and a known model price. The error names the missing condition.
- **Prompt enhancement is slow:** the model was cold. **Warm** it, or raise its
  `keep_alive`; a 27B load can take ~36 s, a 3B a few seconds.
- **Enhancement did nothing:** the role may be unassigned or the provider failed.
  The job keeps the original prompt and records the failed role in
  `Job.params_json.llm`.

---

## 15. Known limitations

- Studio and the legacy editor now use separate document stores against the same
  backend; legacy `/` remains until the final cutover. Set
  `VITE_STUDIO_ENABLED=false` to serve legacy at `/studio`, or
  `VITE_STUDIO_DISABLED_SURFACES=shorts,setup` to hide specific studio surfaces.
- Depth overlay requires the optional Depth Anything V2 Small model; object
  tracking and masks are integrated.
- Tauri provides a native shell with Open/Save dialogs and can launch a
  `milimo-backend` sidecar binary if one is shipped; no Python runtime is bundled
  yet.
- Visual baselines are platform-tolerant; performance is measured in headless
  Chromium and may differ on a desktop compositor.
- Shorts and lenses are scoped F5: candidate ranges are editable, recipes
  import/export, and lens packs (`{name, recipes}`) import with per-recipe
  selection. Shorts template packs remain.
- Scoped v1 cut: only a V1 crossfade is implemented (no dip/wipe/slide
  transitions); LUT, titles, A2, lens pack import, and captions are supported.
  Shorts template packs remain. Search is ranked
  lexical matching, not embeddings.
- The depth model is downloaded separately; it is not shipped with the repo.
- **Memory**: costs are a measured estimate, not a per-model profile; Strata has no
  remote unload; a cancelled job keeps its pipeline until the idle TTL; memory
  waits are bounded (20 retries); swap exhaustion can refuse jobs until the machine
  is restarted. See `Milimo Video/systems/memory-management-audit-2026-11.md`.
- **ComfyUI**: graphs are imported as API-format JSON, not edited in Milimo;
  residency is a declared `MILIMO_COMFY_GB`, not measured per model; a sub-second
  graph can finish before the websocket connects, so completion falls back to
  history polling. ComfyUI defaults to port 8001 and SAM 3 to port 8002.
- **Model efficiency (M5)**: SAM is lazy and `backend/benchmark_models.py` records
  per-lane baselines, but quantized/distilled checkpoints and text-encoder
  alternatives need model artifacts that are not vendored.
- **Notifications**: the legacy `/` editor receives mapped events but has no toast
  layer or activity center. macOS desktop notifications use `osascript`; the Tauri
  shell has a tray icon with show/quit and tooltip progress.
- Phase 6 v2 artifact work (model swaps, MLX/FP8 lanes, agent crew) is a separate
  program and is not part of scoped v1.
