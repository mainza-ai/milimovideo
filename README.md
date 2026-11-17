# Milimo Video

<div align="center">

![Milimo Video](web-app/public/logo.png)

**The AI-Native Cinematic Studio**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Frontend](https://img.shields.io/badge/Frontend-React_19_+_Vite-61DAFB.svg)](web-app/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI_Python-009688.svg)](backend/)
[![Models](https://img.shields.io/badge/Models-LTX--2_|_Flux.2_|_SAM_3-purple.svg)](https://github.com/Lightricks/LTX-2)

</div>

---

**Milimo Video** is an open-source, local-first AI film studio: a storyboard, a
shot-generation workspace, a non-linear editor, a review room, and a delivery desk
in one application. It runs on your machine. No cloud account is required for the
project, the cut, the models, or the render.

There are two interfaces during the strangler cutover:

- **Studio** — `http://localhost:5173/studio`. The current shell: Story, Generate,
  Edit, Review, Library, Deliver, Shorts, and Setup.
- **Legacy editor** — `http://localhost:5173/`. Frozen fallback for the cutover
  window. Set `VITE_STUDIO_ENABLED=false` to serve it at `/studio` as a rollback.

The Tauri desktop shell hosts the same Studio UI with native Open/Save dialogs and
a backend supervisor.

## ✨ What makes it a studio

### One project, three lenses

- **Story** — script parsing, story spine, infinite board, shot list, typed links,
  notes, master-reference cards, review task cards, and board undo.
- **Generate** — ingredients and `@trigger` mentions, advanced parameters, a
  persistent seed lock, start/end-frame conditioning, preview stills, batch queue
  and matrix, immutable takes, favorites, inpaint, and a pipeline lens whose
  reproducible recipe can be exported and imported as JSON.
- **Edit** — program monitor, compare/wipe, guides, magnetic V1 plus free V2/A1,
  trim/slip/roll/blade/ripple/rate/slide, audio gain/mute/fades/solo, waveform,
  captions with SRT/VTT import and burn-in, motion keyframes, render range, a
  timeline table/keyboard path, object tracking, local depth overlay, blocking
  lens, and lazy 3D previz.

### Takes, review, and delivery

- Every completed generation is an immutable take with seed, model, and lineage.
- Review supports frame comments, approvals, activity, compare, a needs-you tray,
  and capability share links.
- Library covers uploads, master references, takes, and ranked local search.
- Deliver writes the computed cut with V1/V2 compositing, embedded audio + A1 mix,
  trims, speed, fades, keyframes, captions, render range, and persisted encoding
  settings (CRF, encoder speed, audio bitrate, optional loudness normalization).
- Shorts builds scored candidate windows from a transcript or shot text, supports
  editable in/out ranges, and exports 9:16, 1:1, and 4:5 crops from the cut or an
  uploaded source.
- Setup exposes real device/slot/queue/disk state, queued and cancellable model
  downloads, LoRA activation and strength, an optional local Depth Anything V2
  model, queue pause/resume, integrity scan/repair, and a job error budget.

### Visual conditioning & character consistency

- **Story Elements / Master References** — reusable Characters, Locations, Objects,
  and Styles with `@trigger` words. Trigger detection injects reference images and
  enriched prompt context.
- **IP-Adapter** — Flux 2 reference conditioning for identity and style.
- **Reference conditioning** — native AE reference encoding for generation.
- **Projected latents** — reference images can be projected into LTX-2 latent space.

### Storyboard engine and chained generation

- Paste screenplay text; the local parser produces scenes and shots. AI parsing is
  queued through the GPU ledger rather than blocking the request.
- Character/Location/Object matching uses `ElementMatcher` with provenance.
- Shots over the native 121-frame chunk generate autoregressively with a 25-frame
  overlap, pixel `media_sequence` handoff, and start-trimmed audio alignment.

### Inpainting, segmentation, tracking, depth

- **Flux 2 RePaint** inpainting from a text mask, SAM points, or an uploaded mask.
- **SAM 3** text/box/click segmentation and bidirectional video object tracking via
  a separate microservice on port 8001.
- **Depth Anything V2 Small** (optional, downloaded in Setup) produces a local
  monocular depth map for the Edit stage.
- Jobs run through the durable GPU ledger with one OOM retry and a single heavy
  model slot.

![Milimo Elements](assets/milimo_video_elements.png)

![Milimo Timeline](assets/milimo_video_timeline.png)

![Milimo Image Generation](assets/milimo_video_image_generation.png)

## 🛠️ Architecture

![Milimo Architecture](assets/Milimo_Video_Studio_Architecture_infograph.png)

| Layer | Technology |
|---|---|
| **Studio frontend** | React 19.2, TypeScript, Vite, Zustand document/editor/job/agent stores, TanStack Query, zod contracts |
| **Legacy frontend** | React + Vite NLE at `/`, separate timeline store, frozen fallback |
| **Backend** | FastAPI (Python 3.12 in the checked-in venv), SQLModel/SQLAlchemy + SQLite, Alembic, SSE via `sse-starlette` |
| **Video AI** | LTX-2 19B — `ti2vid`, `ic_lora`, `keyframe`, chained generation |
| **Image AI** | Flux 2 Klein 9B — IP-Adapter, True CFG, RePaint inpainting |
| **Segmentation** | SAM 3 microservice (port 8001) — text/box/click masks and video tracking |
| **Prompt AI** | Gemma 3 via the LTX text encoder; optional Ollama path |
| **Processing** | FFmpeg/ffprobe — render, probes, thumbnails, frame extraction, stitching |
| **Desktop** | Tauri 2 — native Open/Save dialogs, backend supervisor, sidecar-binary lookup |

### Backend surface

- Eleven routers and 113 routes: projects, jobs, assets, storyboard, elements,
  shots, models, planner, review, share, depth.
- Pydantic write boundaries, generated OpenAPI TypeScript, and zod contract checks.
- Alembic baseline `0001_baseline` plus `0002_render_settings` and
  `0003_seed_lock`; legacy DBs are repaired and stamped at head.
- `QueuedJob` ledger, one global GPU semaphore, one OOM retry, startup recovery,
  idempotent shot jobs, pause/resume dispatch, queue position/estimate.
- `X-Request-ID`, timing logs, `/health`, `/metrics/jobs`, and an error-budget UI.
- Loopback by default, CORS allowlist, optional bearer token, safe Range media
  serving, upload allowlist/size cap.

### MPS-first notes

- FP8 is CUDA-only; Apple Silicon uses the regular MPS path.
- The VAE is cast to float32 on MPS to prevent black output; accelerator caches are
  cleared between jobs.
- Long video jobs are chunked; the GPU semaphore prevents two heavy models loading
  at once.
- SAM 3 auto-selects CUDA → MPS → CPU and guards CUDA-only calls.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is used for unsupported SAM 3 ops.

## 📚 Documentation

- [User Guide](docs/user_guide.md) — the definitive Studio-first manual.
- [Wiki index](Milimo%20Video/index.md) — interlinked systems, models, pipelines,
  workflows, decisions, and roadmap.
- [System Architecture](docs/01_system_architecture.md)
- [Data Models](docs/02_data_models.md)
- [AI Pipelines](docs/03_ai_pipelines.md)
- [Frontend State](docs/04_frontend_state.md)
- [Execution Flow](docs/05_execution_flow.md)
- [File Dependencies](docs/06_file_dependency.md)
- [Flux 2 Bible](docs/flux2-bible.md), [LTX-2 Bible](docs/ltx2-bible.md),
  [SAM 3 Bible](docs/sam3-bible.md)
- [Implementation plan](Milimo%20Video/roadmap/implementation-plan-2026-11.md) —
  scoped v1 status, remaining/deferred items, verification matrix.

## 🚀 Getting Started

### Prerequisites

- **Python 3.12** (the repo venv is `milimov/`)
- **Node.js 22+**
- **FFmpeg and ffprobe** on `PATH`
- **Apple Silicon:** 32 GB unified memory or more recommended for LTX-2 19B.
- **NVIDIA:** 16 GB+ VRAM recommended.
- **Disk:** model checkpoints are large; Setup reports free space before download.

### 1. Install

```bash
git clone https://github.com/mainza-ai/milimovideo.git
cd milimovideo
python3 -m venv milimov
./milimov/bin/pip install -e ./LTX-2/packages/ltx-core \
  -e ./LTX-2/packages/ltx-pipelines -e ./flux2 \
  -r backend/requirements.txt
cd web-app && npm install && cd ..
```

### 2. Models

Download LTX-2, Flux 2, and SAM 3 checkpoints as described in the
[User Guide](docs/user_guide.md), or use Setup's model table to inspect/download
supported entries. Depth Anything V2 Small is optional and only needed for the
Edit depth overlay.

SAM 3 is optional for segmentation/tracking; without it the rest of the studio
still runs.

### 3. Launch

```bash
./run_backend.sh     # verifies the environment, then serves http://127.0.0.1:8000
./run_sam.sh         # optional SAM 3 on port 8001
./run_frontend.sh    # Vite on http://127.0.0.1:5173
```

Open **`http://localhost:5173/studio`** for the Studio shell. The legacy editor is
at `/`.

Desktop:

```bash
cd web-app
npm run tauri dev
npm run tauri build
```

If the backend is offline, the top-bar status offers **Start backend**, which runs
`run_backend.sh` from the checkout. No Python/Torch runtime is bundled yet; the
supervisor will launch a `milimo-backend` sidecar binary if one is shipped.

### Health

```bash
curl http://127.0.0.1:8000/health
```

Reports device, GPU busy/queued, memory slots, queue pause state, auth mode, and
free disk.

## 🎮 Controls

### Studio

| Key | Action |
|---|---|
| `1`–`8` | Story, Generate, Edit, Review, Library, Deliver, Shorts, Setup |
| `Space` | Play/pause (opens Edit when needed) |
| `⌘K` / `Ctrl+K` | Command palette |
| `⌘J` / `Ctrl+J` | Agent command dock |
| `?` | Keyboard shortcuts |
| `⌘Z` / `Ctrl+Z` | Undo |
| `⇧⌘Z` / `Ctrl+Shift+Z` | Redo |
| `B` | Blade at the playhead |
| `R` | Ripple delete |
| `[` / `]` | Slide selected V1 clip |
| `Escape` | Exit focus mode |

Board: drag empty canvas to pan, `⌘`/`Ctrl`+wheel to zoom, `Shift`-click to
multi-select, `Shift`-drag to marquee. Timeline: drag clip edges to trim; the table
view exposes keyboard editing.

### Legacy

The legacy editor keeps its original shortcuts, including `S`/`⌘S` save and the
masking/tracking toggles. New work belongs in Studio.

## ✅ Verification

```bash
./milimov/bin/python -m compileall -q backend
./milimov/bin/python -m pytest                    # 82 tests
./milimov/bin/python -m ruff check backend        # correctness rules
./milimov/bin/python backend/verify_setup.py      # 0 failures expected

cd web-app
npx tsc -b --pretty false
npm run lint
npm test                                          # 53 tests
npm run build
npm run check:bundle                              # JS budget
npm run test:e2e                                  # 15 Playwright tests
```

The last verified run was 82 backend tests, 53 Vitest tests, and 15 Playwright
tests, including all-surface axe, CRUD, export, reference CRUD, review-agent,
responsive layout/overflow, perf, and visual checks.

## 🧭 Project layout

```
backend/           FastAPI, tasks, routes, models, Alembic, tests
web-app/           React/Vite Studio + legacy editor + Tauri shell
  src/studio/      Studio shell, stores, surfaces, lenses, shorts, collab
  src-tauri/      Native shell and backend supervisor
sam3/              Optional SAM 3 microservice
tests/             Backend pytest suite
docs/              Immutable technical sources and user guide
Milimo Video/      LLM-owned wiki (systems, models, pipelines, roadmap)
dev_diary/         Build log and social notes
```

## ⚠️ Known limitations

- **No bundled Python runtime.** The desktop shell can start a sidecar binary if
  present, but packaging Python/Torch is not built.
- **Legacy `/` remains** until the documented two-clean-release cutover window.
- **Scoped v1 cut:** no transitions, LUTs, titles, or A2 track; search is ranked
  lexical matching, not embeddings; Shorts/lenses have no template packs.
- **Depth model** is downloaded separately and is not shipped in the repo.
- **Visual baselines** are platform-tolerant and locally generated.
- **Phase 6 v2 work** (eval harness, model swaps, MLX/FP8 lanes, agent crew) is a
  separate program, not part of scoped v1.

## 🤝 Community

Milimo Video is open source. Contributions are welcome:

- **Code** — pull requests for features, fixes, and performance.
- **Models** — share LoRAs or fine-tunes that work with LTX-2/Flux.
- **Feedback** — open issues on the [GitHub repository](https://github.com/mainza-ai/milimovideo).

## ✍️ Author

**Mainza Kangombe**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/mainza-kangombe-6214295)

## 📜 License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
