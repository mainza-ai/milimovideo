"""MPS weight loading uses the measured pread backend when available."""

from __future__ import annotations

import torch

from ltx_core.loader.sft_loader import SafetensorsStateDictLoader


class _FakeHandle:
    def __init__(self):
        self._tensor = torch.zeros(2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return ["weight"]

    def get_tensor(self, _name):
        return self._tensor


def test_loader_selects_pread_on_mps(monkeypatch):
    calls = {}

    def fake_open(path, framework, device, *, backend=None):
        calls.update({"path": path, "device": device, "backend": backend})
        return _FakeHandle()

    import ltx_core.loader.sft_loader as module
    monkeypatch.setattr(module.safetensors, "safe_open", fake_open)
    loader = SafetensorsStateDictLoader()
    loader.load("weights.safetensors", sd_ops=None, device=torch.device("mps"))
    assert calls["backend"] == "pread"
    assert calls["device"] == "mps"
