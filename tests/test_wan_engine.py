"""Wan 2.2 adapter: cost, routing, normalization, and pipeline contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest


def test_wan_cost_is_lower_than_the_ltx_base():
    from services.memory import costs

    wan = costs.estimate("video", {"engine": "wan", "num_frames": 49, "width": 480, "height": 480})
    ltx = costs.estimate("video", {"num_frames": 49, "width": 480, "height": 480})
    assert wan.base_gb == costs.WAN_BASE_GB
    assert wan.total_gb < ltx.total_gb
    assert wan.notes and "Wan 2.2" in wan.notes[0]


def test_wan_manifest_entry():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "backend/models_manifest.json").read_text())
    entry = next(item for item in data["models"] if item["id"] == "wan2.2-ti2v-5b")
    assert entry["huggingface"]["repo"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert entry["huggingface"]["subdir"] == ""
    assert entry["license"] == "apache-2.0"


def test_wan_normalizes_frames_and_resolution():
    from models.wan_wrapper import WanEngine

    assert WanEngine.normalize(49, 480, 480) == (49, 480, 480)
    assert WanEngine.normalize(48, 481, 481) == (49, 480, 480)
    frames, width, height = WanEngine.normalize(121, 1280, 704)
    assert frames == 121
    assert width % 16 == 0 and height % 16 == 0


def test_wan_engine_loads_and_generates_through_diffusers(monkeypatch, tmp_path):
    import diffusers
    import torch

    from models.wan_wrapper import WanEngine

    calls = {}

    class FakePipe:
        def to(self, device):
            calls["device"] = device
            return self

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return type("Out", (), {"frames": [np.zeros((49, 480, 480, 3), dtype=np.float32)]})()

    class FakeWanPipeline:
        @classmethod
        def from_pretrained(cls, model_dir, torch_dtype=None):
            calls["model_dir"] = model_dir
            calls["dtype"] = torch_dtype
            return FakePipe()

    monkeypatch.setattr(diffusers, "WanPipeline", FakeWanPipeline)
    (tmp_path / "model_index.json").write_text("{}")
    engine = WanEngine(str(tmp_path), "cpu")
    engine.load()
    frames, count = engine.generate("a cat", width=480, height=480, num_frames=49, seed=1)
    assert count == 49
    assert frames.shape[0] == 49
    assert calls["device"] == "cpu"
    assert calls["generate"]["num_frames"] == 49
    assert calls["generate"]["output_type"] == "np"


@pytest.mark.asyncio
async def test_video_task_routes_wan_jobs(app_client, monkeypatch):
    from sqlmodel import Session

    from database import Job, engine
    from tasks import video as video_task
    from tasks import wan as wan_task

    with Session(engine) as session:
        session.add(Job(id="job-wan-route", type="video", status="processing"))
        session.commit()

    calls = {}

    async def fake(job_id, params):
        calls["job_id"] = job_id
        calls["params"] = params

    monkeypatch.setattr(wan_task, "generate_wan_video_task", fake)
    await video_task.generate_video_task(
        "job-wan-route", {"pipeline_override": "wan", "prompt": "x"}
    )
    assert calls["job_id"] == "job-wan-route"
    assert calls["params"]["pipeline_override"] == "wan"
