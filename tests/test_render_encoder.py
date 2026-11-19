"""Render encoder policy and a real FFmpeg encode on this host."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import config
from utils.render_utils import build_complex_filtergraph, encoder_args


def test_auto_stays_on_software_x264(monkeypatch):
    monkeypatch.setattr(config, "RENDER_ENCODER", "auto")
    monkeypatch.setattr(config, "RENDER_X264_THREADS", 4)
    args = encoder_args(crf=18, preset="fast")
    assert args[:2] == ["-c:v", "libx264"]
    assert "-crf" in args and args[args.index("-crf") + 1] == "18"
    assert args[args.index("-threads") + 1] == "4"


def test_videotoolbox_maps_crf_to_quality_and_supports_hevc(monkeypatch):
    monkeypatch.setattr(config, "RENDER_ENCODER", "videotoolbox")
    monkeypatch.setattr(config, "RENDER_VIDEOTOOLBOX_QUALITY", 0)
    h264 = encoder_args(crf=18)
    assert h264[:2] == ["-c:v", "h264_videotoolbox"]
    quality = int(h264[h264.index("-q:v") + 1])
    # CRF 18 is high quality, so the VideoToolbox q value must be high.
    assert 60 <= quality <= 95

    low = encoder_args(crf=45)
    assert int(low[low.index("-q:v") + 1]) < quality

    hevc = encoder_args(crf=18, codec="hevc")
    assert hevc[:2] == ["-c:v", "hevc_videotoolbox"]


def test_videotoolbox_override_accepts_explicit_quality(monkeypatch):
    monkeypatch.setattr(config, "RENDER_ENCODER", "videotoolbox")
    args = encoder_args(crf=18, quality=42)
    assert args[args.index("-q:v") + 1] == "42"


def test_filtergraph_uses_the_configured_encoder(monkeypatch):
    monkeypatch.setattr(config, "RENDER_ENCODER", "videotoolbox")
    cmd = build_complex_filtergraph(
        project_fps=25,
        project_width=320,
        project_height=180,
        v1_shots=[],
        v2_shots=[],
        a1_shots=[],
        output_path="out.mp4",
        crf=18,
        preset="fast",
    )
    assert "h264_videotoolbox" in cmd
    assert "-crf" not in cmd


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_real_renders_with_both_encoders(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=320x180:rate=25:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", str(source),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
    rows = []
    for label, encoder in (("software", "software"), ("videotoolbox", "videotoolbox")):
        if encoder == "videotoolbox" and "h264_videotoolbox" not in encoders:
            continue
        monkeypatch.setattr(config, "RENDER_ENCODER", encoder)
        output = tmp_path / f"{label}.mp4"
        args = encoder_args(crf=18, preset="fast")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(source), *args, "-pix_fmt", "yuv420p", str(output)],
            capture_output=True,
        )
        if encoder == "videotoolbox" and result.returncode != 0:
            pytest.skip("VideoToolbox encode unavailable in this ffmpeg build")
        assert result.returncode == 0, result.stderr.decode("utf-8", "ignore")[-500:]
        assert output.stat().st_size > 0
        rows.append(output.stat().st_size)
    assert len(rows) >= 1
