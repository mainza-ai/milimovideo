---
type: workflows
created: 2026-10-09
updated: 2026-10-09
tags: setup installation quickstart
sources:
  - README.md
  - docs/user_guide.md
related:
  - ../overview.md
  - ../concepts/memory-slots.md
---

# Getting Started

## Goal

From a fresh clone to a running studio generating video, locally.

## Prerequisites

- Python 3.10+, Node.js 18+, FFmpeg
- **NVIDIA**: 16GB+ VRAM (RTX 3090/4090 recommended)
- **Apple Silicon**: 32GB+ unified memory (M2/M3/M4 Max/Ultra recommended)

## Steps

1. **Clone**: `git clone https://github.com/mainza-ai/milimovideo.git && cd milimovideo`
2. **Backend env**: create venv `milimov`, install `./LTX-2/packages/ltx-core`,
   `./LTX-2/packages/ltx-pipelines`, `./flux2`, then `backend/requirements.txt`.
3. **LTX-2 checkpoints** → `LTX-2/models/checkpoints/`: the 19B distilled
   safetensors, the distilled LoRA-384, the spatial upscaler → `upscalers/`; Gemma 3
   text encoder → `text_encoders/gemma3/`.
4. **Flux 2 weights** → `backend/models/flux2/`: `flux-2-klein-9b.safetensors`,
   `ae.safetensors` (native AE, preferred), `vae/` (diffusers fallback),
   `text_encoder/` (Qwen 3 8B), `tokenizer/`, `ip-adapter.safetensors`.
5. **SAM 3** (optional but recommended): `conda create -n sam3_env python=3.12`,
   `pip install -e sam3` + FastAPI/uvicorn deps; checkpoint `backend/models/sam3/sam3.pt`
   auto-downloads from HuggingFace on first start if missing.
6. **Launch all three services**:
   - `./run_backend.sh` → API on port 8000
   - `./run_sam.sh` → SAM 3 on port 8001
   - `./run_frontend.sh` → UI on port 5173
7. **Open** `http://localhost:5173` (Chrome or Safari). Paste a script into the
   Storyboard view, generate thumbnails for pre-viz, then generate shots.

## Tips

- Only one heavy model lives in memory at a time — switching from image work to
  video work visibly unloads/reloads models (see [[../concepts/memory-slots.md]]).
  Don't run two generations at once.
- On Apple Silicon: if output frames come back black, check the MPS dtype guards
  (VAE decode on CPU, float32 transformer).
- Inpainting and tracking need the SAM 3 service running; generation alone doesn't.
- Keyboard: `Space` play/pause, `Cmd+S` save, drag-and-drop imports media.

## Related

- [[../overview.md]]
- [[../concepts/memory-slots.md]]
