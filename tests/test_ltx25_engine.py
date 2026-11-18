"""LTX-2.5 adapter contract (weights are gated; the pipeline is faked)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class _FakePipe:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        frames = np.zeros((3, 32, 32, 3), dtype=np.float32)
        return type("Out", (), {"frames": [frames], "audio": None})()


def test_ltx25_normalizes_frames_and_resolution():
    from models.ltx25_wrapper import LTX25Engine

    engine = LTX25Engine("/missing", "cpu")
    assert engine.normalize(120, 769, 513) == (121, 768, 512)
    assert engine.normalize(49, 480, 480) == (49, 480, 480)


def test_ltx25_generate_uses_the_diffusers_contract():
    from models.ltx25_wrapper import LTX25Engine

    engine = LTX25Engine("/missing", "cpu")
    pipe = _FakePipe()
    engine.pipe = pipe
    steps = []
    frames, audio, count = engine.generate(
        "a storm over the sea",
        "blurry",
        width=320,
        height=256,
        num_frames=49,
        fps=25,
        num_inference_steps=8,
        guidance=4.0,
        seed=7,
        callback=lambda step, total: steps.append((step, total)),
    )
    assert frames.shape == (3, 32, 32, 3)
    assert count == 49
    assert audio is None
    assert pipe.kwargs["width"] == 320
    assert pipe.kwargs["height"] == 256
    assert pipe.kwargs["num_frames"] == 49
    assert pipe.kwargs["frame_rate"] == 25.0
    assert pipe.kwargs["num_inference_steps"] == 8
    assert steps == []


def test_ltx25_is_recommended_and_gated_in_the_manifest():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "backend/models_manifest.json").read_text())
    entry = next(item for item in data["models"] if item["id"] == "ltx25-diffusers")
    assert entry["family"] == "ltx25"
    assert entry["gated"] is True
    assert entry["recommended"] is True
    assert entry["huggingface"]["repo"] == "Lightricks/LTX-2.5-Diffusers"


def test_ltx25_cost_uses_a_larger_video_budget():
    from services.memory import costs

    ltx25 = costs.estimate("video", {"engine": "ltx25", "num_frames": 49, "width": 480, "height": 480})
    wan = costs.estimate("video", {"engine": "wan", "num_frames": 49, "width": 480, "height": 480})
    assert ltx25.base_gb == costs.LTX25_BASE_GB
    assert ltx25.total_gb > wan.total_gb
    assert "LTX-2.5" in ltx25.notes[0]


def test_ltx25_pipeline_override_dispatches_to_the_adapter(app_client, monkeypatch):
    import asyncio

    from tasks import video as video_task

    called = {}

    async def fake(job_id, params):
        called["job_id"] = job_id
        called["params"] = params

    monkeypatch.setattr("tasks.ltx25.generate_ltx25_video_task", fake)
    asyncio.run(video_task.generate_video_task(
        "ltx25-dispatch",
        {"pipeline_override": "ltx25", "enhance_prompt": False, "prompt": "x"},
    ))
    assert called["job_id"] == "ltx25-dispatch"
    assert called["params"]["pipeline_override"] == "ltx25"
