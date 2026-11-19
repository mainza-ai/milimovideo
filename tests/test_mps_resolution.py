"""Frame-aware MPS resolution clamp."""

from __future__ import annotations

import pytest

import config
from tasks import video


@pytest.fixture()
def mps(monkeypatch):
    monkeypatch.setattr(video.torch.backends.mps, "is_available", lambda: True)


def test_short_clip_keeps_the_reference_budget(mps):
    width, height = video.clamp_resolution_for_device(1280, 768, num_frames=121)
    assert (width, height) == (1280, 768)


def test_long_clip_scales_below_the_reference_ceiling(mps):
    long_width, long_height = video.clamp_resolution_for_device(1280, 768, num_frames=481)
    assert long_width * long_height < config.MPS_MAX_PIXELS
    assert long_width % 64 == 0
    assert long_height % 64 == 0
    short_width, short_height = video.clamp_resolution_for_device(1280, 768, num_frames=121)
    assert long_width < short_width
    assert long_height < short_height


def test_oversized_frames_clamp_proportionally_and_align(mps):
    width, height = video.clamp_resolution_for_device(1920, 1080, num_frames=121)
    assert width * height <= config.MPS_MAX_PIXELS
    assert width % 64 == 0
    assert abs((width / height) - (1920 / 1080)) < 0.1


def test_cpu_and_cuda_are_not_clamped(monkeypatch):
    monkeypatch.setattr(video.torch.backends.mps, "is_available", lambda: False)
    assert video.clamp_resolution_for_device(1920, 1080, num_frames=481) == (1920, 1080)
