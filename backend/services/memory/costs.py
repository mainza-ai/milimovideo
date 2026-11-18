"""Measured model costs and per-job peak estimates.

The disk size is a starting point, not a budget. Runtime adds the text encoder,
VAE, upscalers, attention workspace, and the second upscale pass.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

import config

from .models import CostBreakdown

# Measured on the reference 128 GB Apple Silicon machine.
VIDEO_BASE_GB = 76.0        # LTX-2 19B + Gemma 3 + VAE (Apple unified memory)
IMAGE_BASE_GB = 36.0        # Flux 2 9B + Qwen3 encoder + AE/CLIP/IP
# CUDA auto-selects the FP8 LTX checkpoint (manifest: 40 GB) and bf16 Flux 9B.
VIDEO_BASE_GB_CUDA = 40.0
IMAGE_BASE_GB_CUDA = 18.0
# Wan 2.2 TI2V-5B: 5B DiT + UMT5 encoder + high-compression VAE.
# Measured disk/loading footprint: 5B DiT + UMT5 + VAE (fp16), ~34 GB unified;
# CUDA fp16 components fit ~18 GB before workspace.
WAN_BASE_GB = 34.0
WAN_BASE_GB_CUDA = 18.0
# LTX-2.5 22B + text encoder + VAE/audio VAE. Official floor is 32 GB FP8;
# bf16 with CPU offload/VAE tiling is the practical Apple/CUDA path.
LTX25_BASE_GB = 64.0
LTX25_BASE_GB_CUDA = 40.0
SAM_BASE_GB = 8.0
DEPTH_BASE_GB = 0.6

VIDEO_ENCODE_GB = 2.0
IMAGE_SAVE_GB = 1.0
SAFETY_FACTOR = 1.05

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_BITS_RE = re.compile(r"(iq(\d+)_(?:xxs|xs|s|m|l)|q(\d+)(?:_[a-z0-9]+)?|(\d+)\s*bit|bf16|fp16|f16)", re.IGNORECASE)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def base_gb(kind: str) -> float:
    if _cuda_available():
        return {
            "video": VIDEO_BASE_GB_CUDA,
            "image": IMAGE_BASE_GB_CUDA,
            "sam": SAM_BASE_GB,
            "depth": DEPTH_BASE_GB,
        }.get(kind, 0.0)
    return slot_base_gb(kind)


def slot_base_gb(slot: str) -> float:
    return {
        "video": VIDEO_BASE_GB,
        "image": IMAGE_BASE_GB,
        "sam": SAM_BASE_GB,
        "depth": DEPTH_BASE_GB,
        "comfy": float(getattr(config, "COMFY_DECLARED_GB", 2.0)),
    }.get(slot, 0.0)


def _params_billions(text: str) -> float | None:
    matches = _PARAM_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _bits_per_weight(text: str) -> float:
    blob = (text or "").lower()
    match = _BITS_RE.search(blob)
    if not match:
        return 16.0
    if match.group(1):
        return 4.25 if "iq4" in match.group(1) else 4.5
    if match.group(4):
        return float(match.group(4))
    if match.group(5):
        return float(match.group(5))
    return 16.0


def estimate_llm_gb(model_id: str, params_b: float | None = None, bits: float | None = None) -> float:
    """Conservative LLM residency from a name or explicit metadata."""
    params = params_b or _params_billions(model_id) or 8.0
    weight_bits = bits or _bits_per_weight(model_id)
    weights = params * weight_bits / 8.0
    # KV cache, activations, tokenizer/runtime overhead.
    return round(weights * 1.20 + 2.0, 2)


def estimate(kind: str, params: Dict[str, Any] | None = None) -> CostBreakdown:
    params = params or {}
    kind = (kind or "video").lower()
    if kind == "video" and str(params.get("engine") or "").lower() == "ltx25":
        base = LTX25_BASE_GB_CUDA if _cuda_available() else LTX25_BASE_GB
        frames = int(params.get("num_frames") or 121)
        width = int(params.get("width") or 768)
        height = int(params.get("height") or 512)
        scale = max(0.1, (frames / 121.0) * ((width * height) / float(768 * 512)))
        workspace = 4.0 + 10.0 * scale
        return CostBreakdown(
            kind=kind,
            base_gb=base,
            workspace_gb=workspace,
            total_gb=round((base + workspace) * SAFETY_FACTOR, 2),
            frames=frames,
            width=width,
            height=height,
            notes=[f"LTX-2.5 22B, {frames} frames at {width}x{height}"],
        )
    if kind == "video" and str(params.get("engine") or "").lower() == "wan":
        base = WAN_BASE_GB_CUDA if _cuda_available() else WAN_BASE_GB
        frames = int(params.get("num_frames") or 49)
        width = int(params.get("width") or 480)
        height = int(params.get("height") or 480)
        scale = max(0.1, (frames / 49.0) * ((width * height) / float(480 * 480)))
        workspace = 2.0 + 4.0 * scale
        total = (base + workspace) * SAFETY_FACTOR
        return CostBreakdown(
            kind=kind,
            base_gb=base,
            workspace_gb=workspace,
            total_gb=round(total, 2),
            frames=frames,
            width=width,
            height=height,
            notes=[f"Wan 2.2 5B, {frames} frames at {width}x{height}"],
        )
    if kind == "video":
        video_base = base_gb("video")
        frames = int(params.get("num_frames") or 121)
        width = int(params.get("width") or 768)
        height = int(params.get("height") or 512)
        upscale = bool(params.get("upscale"))
        scale = max(0.05, (frames / 121.0) * ((width * height) / float(768 * 512)))
        workspace = 3.0 + 10.0 * scale
        if upscale:
            workspace = workspace * 1.6 + 2.0
        workspace += VIDEO_ENCODE_GB
        total = (video_base + workspace) * SAFETY_FACTOR
        notes = [
            f"{frames} frames at {width}x{height}",
            "upscale pass included" if upscale else "no upscale",
        ]
        return CostBreakdown(
            kind=kind,
            base_gb=video_base,
            workspace_gb=workspace,
            total_gb=round(total, 2),
            upscale=upscale,
            frames=frames,
            width=width,
            height=height,
            notes=notes,
        )
    if kind == "image":
        image_base = base_gb("image")
        refs = int(params.get("reference_count") or 0)
        workspace = 4.0 + min(4, refs) * 0.5 + IMAGE_SAVE_GB
        total = (image_base + workspace) * SAFETY_FACTOR
        return CostBreakdown(
            kind=kind,
            base_gb=image_base,
            workspace_gb=workspace,
            total_gb=round(total, 2),
            frames=1,
            notes=[f"{refs} reference image(s)"],
        )
    if kind == "sam":
        total = SAM_BASE_GB * SAFETY_FACTOR
        return CostBreakdown(kind=kind, base_gb=SAM_BASE_GB, workspace_gb=0.0, total_gb=round(total, 2))
    if kind == "comfy":
        # ComfyUI is an external process with its own model cache. The declared
        # residency covers its cache + the workflow workspace; it is user-tunable
        # because ComfyUI exposes no per-model accounting.
        declared = float(params.get("estimate_gb") or getattr(config, "COMFY_DECLARED_GB", 2.0))
        return CostBreakdown(
            kind=kind,
            base_gb=declared,
            workspace_gb=1.0,
            total_gb=round((declared + 1.0) * SAFETY_FACTOR, 2),
            notes=["ComfyUI external workflow; declared residency " f"{declared:.1f} GB"],
        )
    if kind == "depth":
        total = DEPTH_BASE_GB * SAFETY_FACTOR
        return CostBreakdown(kind=kind, base_gb=DEPTH_BASE_GB, workspace_gb=0.0, total_gb=round(total, 2))
    if kind == "llm":
        estimated = float(params.get("estimate_gb") or estimate_llm_gb(str(params.get("model") or "")))
        return CostBreakdown(
            kind=kind,
            base_gb=estimated,
            workspace_gb=0.0,
            total_gb=round(estimated * SAFETY_FACTOR, 2),
            notes=[f"model {params.get('model') or 'unknown'}"],
        )
    total = min(4.0, float(params.get("estimate_gb") or 1.0))
    return CostBreakdown(kind=kind, base_gb=total, workspace_gb=0.0, total_gb=total)


def file_size_gb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1e9, 2)
    except OSError:
        return 0.0
