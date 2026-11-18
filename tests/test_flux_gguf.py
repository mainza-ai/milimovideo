"""Quantized Flux selection and guards (no model load)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_manifest_has_the_gguf_checkpoint():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "backend/models_manifest.json").read_text())
    entries = {entry["id"]: entry for entry in data["models"]}
    entry = entries["flux2-klein-9b-q4km"]
    assert entry["type"] == "quantized"
    assert entry["huggingface"]["repo"] == "unsloth/FLUX.2-klein-9B-GGUF"
    assert entry["huggingface"]["filename"].endswith("Q4_K_M.gguf")
    assert entry["size_bytes"] == 5673900320


def test_inpainter_uses_gguf_and_guards_unsupported_features(tmp_path, monkeypatch):
    import config
    from models import flux_wrapper

    gguf_file = tmp_path / "flux.gguf"
    gguf_file.write_bytes(b"gguf")

    calls = {}

    class FakeGGUF:
        def __init__(self, model_dir, path, device, offload=False):
            calls["init"] = (model_dir, path, device, offload)

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return "IMAGE"

        def unload(self):
            calls["unload"] = True

    monkeypatch.setattr(config, "FLUX_GGUF_PATH", str(gguf_file))
    monkeypatch.setattr(config, "FLUX_GGUF_OFFLOAD", True)

    import models.flux_gguf as flux_gguf_module
    monkeypatch.setattr(flux_gguf_module, "FluxGGUF", FakeGGUF)

    from memory_manager import memory_manager
    monkeypatch.setattr(memory_manager, "prepare_for", lambda *a, **k: None)
    monkeypatch.setattr(memory_manager, "release", lambda *a, **k: None)

    inpainter = flux_wrapper.FluxInpainter(device="mps")
    inpainter.load_model()
    assert inpainter.quantized is True
    assert calls["init"][1] == str(gguf_file)

    assert inpainter.generate_image(
        prompt="a key", width=256, height=256, guidance=1.0,
        num_inference_steps=2, seed=7,
    ) == "IMAGE"
    assert calls["generate"]["width"] == 256
    assert calls["generate"]["seed"] == 7

    with pytest.raises(RuntimeError, match="IP-Adapter"):
        inpainter.generate_image(prompt="x", ip_adapter_images=["ref.png"])
    with pytest.raises(RuntimeError, match="inpainting"):
        inpainter.inpaint(None, None, "x")

    inpainter.unload()
    assert calls.get("unload") is True
    assert inpainter.quantized is False
