import json
import shutil
import subprocess
from pathlib import Path

import pytest

import config


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _make_audio(path: Path, seconds: float = 2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={seconds}",
            "-c:a", "aac", "-b:a", "128k", str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _make_clip(path: Path, color: str, seconds: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=25:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _average_rgb(path: Path, seconds: float):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    pixels = len(raw) // 3
    return tuple(
        sum(raw[channel::3]) / max(1, pixels)
        for channel in range(3)
    )


def test_render_pixels_follow_computed_timeline(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "pixel-red.mp4", "red")
    _make_clip(generated / "pixel-blue.mp4", "blue")

    shots = []
    for index, name in enumerate(("red", "blue")):
        shots.append({
            "id": f"pixel-{name}", "prompt": name, "negative_prompt": "", "seed": 42,
            "width": 320, "height": 180, "fps": 25, "num_frames": 25,
            "cfg_scale": 1.0, "enhance_prompt": False, "upscale": False,
            "pipeline_override": "auto", "track_index": 0, "start_frame": 0,
            "trim_in": 0, "trim_out": 0, "timeline": [],
            "video_url": f"/projects/{pid}/generated/pixel-{name}.mp4",
            "status": "completed", "sequence_index": index,
        })
    payload = {
        "id": pid, "name": "Pixel Render", "fps": 25, "seed": 42,
        "resolution_w": 320, "resolution_h": 180, "revision": 0,
        "shots": shots, "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    response = app_client.post(
        f"/projects/{pid}/render",
        json={"settings": {"width": 320, "height": 180, "fps": 25}, "track_states": {}},
    )
    render_id = response.json()["job_id"]
    status = app_client.get(f"/status/{render_id}").json()
    assert status["status"] == "completed", status

    output = generated / f"{render_id}.mp4"
    first = _average_rgb(output, 0.25)
    second = _average_rgb(output, 1.75)
    assert first[0] > first[2] + 50, first
    assert second[2] > second[0] + 50, second


def test_render_v2_alpha_blends_pixels(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "v1-red.mp4", "red")
    _make_clip(generated / "v2-blue.mp4", "blue")

    def shot(name, track, opacity=None):
        data = {
            "id": f"alpha-{name}", "prompt": name, "negative_prompt": "", "seed": 42,
            "width": 320, "height": 180, "fps": 25, "num_frames": 25,
            "cfg_scale": 1.0, "enhance_prompt": False, "upscale": False,
            "pipeline_override": "auto", "track_index": track,
            "start_frame": 0, "trim_in": 0, "trim_out": 0, "timeline": [],
            "video_url": f"/projects/{pid}/generated/{name}.mp4",
            "status": "completed",
        }
        if opacity is not None:
            data["keyframes"] = [
                {"frame": 0, "opacity": opacity},
                {"frame": 24, "opacity": opacity},
            ]
        return data

    payload = {
        "id": pid, "name": "Alpha Render", "fps": 25, "seed": 42,
        "resolution_w": 320, "resolution_h": 180, "revision": 0,
        "shots": [
            {**shot("v1-red", 0), "sequence_index": 0},
            {**shot("v2-blue", 1, 0.5), "sequence_index": 1},
        ],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    response = app_client.post(
        f"/projects/{pid}/render",
        json={"settings": {"width": 320, "height": 180, "fps": 25}, "track_states": {}},
    )
    render_id = response.json()["job_id"]
    status = app_client.get(f"/status/{render_id}").json()
    assert status["status"] == "completed", status

    rgb = _average_rgb(generated / f"{render_id}.mp4", 0.5)
    assert rgb[0] > 40 and rgb[2] > 40, rgb


def test_render_from_an_explicit_source_path(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "source.mp4", "orange", seconds=2.0)

    payload = {
        "id": pid, "name": "Source Render", "fps": 25, "seed": 42,
        "resolution_w": 320, "resolution_h": 180, "revision": 0,
        "shots": [], "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    response = app_client.post(
        f"/projects/{pid}/render",
        json={
            "settings": {
                "width": 180, "height": 320, "fps": 25, "fit": "crop",
                "start_frame": 25, "end_frame": 50,
            },
            "track_states": {},
            "source_path": f"/projects/{pid}/generated/source.mp4",
        },
    )
    assert response.status_code == 200, response.text
    render_id = response.json()["job_id"]
    status = app_client.get(f"/status/{render_id}").json()
    assert status["status"] == "completed", status

    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-show_entries", "stream=width,height", "-select_streams", "v:0",
        "-of", "json", str(generated / f"{render_id}.mp4"),
    ]))
    assert float(probe["format"]["duration"]) == pytest.approx(1.0, abs=0.15)
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (180, 320)


def test_render_crop_fits_a_vertical_frame(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "crop.mp4", "green")
    shot = {
        "id": f"crop-shot-{pid}", "prompt": "crop", "negative_prompt": "", "seed": 42,
        "width": 320, "height": 180, "fps": 25, "num_frames": 25,
        "cfg_scale": 1.0, "enhance_prompt": False, "upscale": False,
        "pipeline_override": "auto", "track_index": 0, "start_frame": 0,
        "trim_in": 0, "trim_out": 0, "timeline": [],
        "video_url": f"/projects/{pid}/generated/crop.mp4", "status": "completed",
    }
    payload = {
        "id": pid, "name": "Crop Render", "fps": 25, "seed": 42,
        "resolution_w": 320, "resolution_h": 180, "revision": 0,
        "shots": [shot], "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    response = app_client.post(
        f"/projects/{pid}/render",
        json={"settings": {
            "width": 180, "height": 320, "fps": 25, "fit": "crop",
        }, "track_states": {}},
    )
    render_id = response.json()["job_id"]
    status = app_client.get(f"/status/{render_id}").json()
    assert status["status"] == "completed", status

    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json",
        str(generated / f"{render_id}.mp4"),
    ]))
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"]) == (180, 320)


def test_render_range_exports_a_subsection(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "range.mp4", "purple", seconds=2.0)

    shot = {
        "id": "range-shot", "prompt": "range", "negative_prompt": "", "seed": 42,
        "width": 320, "height": 180, "fps": 25, "num_frames": 50,
        "cfg_scale": 1.0, "enhance_prompt": False, "upscale": False,
        "pipeline_override": "auto", "track_index": 0, "start_frame": 0,
        "trim_in": 0, "trim_out": 0, "timeline": [],
        "video_url": f"/projects/{pid}/generated/range.mp4", "status": "completed",
    }
    payload = {
        "id": pid, "name": "Range Render", "fps": 25, "seed": 42,
        "resolution_w": 320, "resolution_h": 180, "revision": 0,
        "shots": [shot], "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    response = app_client.post(
        f"/projects/{pid}/render",
        json={
            "settings": {
                "width": 320, "height": 180, "fps": 25,
                "start_frame": 25, "end_frame": 50,
            },
            "track_states": {},
        },
    )
    assert response.status_code == 200, response.text
    render_id = response.json()["job_id"]
    status = app_client.get(f"/status/{render_id}").json()
    assert status["status"] == "completed", status

    output = generated / f"{render_id}.mp4"
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(output),
    ]))
    assert float(probe["format"]["duration"]) == pytest.approx(1.0, abs=0.15)


def test_render_speed_halves_the_cut(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "speed.mp4", "orange", seconds=2.0)

    shot = {
        "id": "speed-render-shot",
        "prompt": "speed",
        "negative_prompt": "",
        "seed": 42,
        "width": 320,
        "height": 180,
        "fps": 25,
        "num_frames": 50,
        "speed": 2.0,
        "cfg_scale": 1.0,
        "enhance_prompt": False,
        "upscale": False,
        "pipeline_override": "auto",
        "track_index": 0,
        "start_frame": 0,
        "trim_in": 0,
        "trim_out": 0,
        "timeline": [],
        "video_url": f"/projects/{pid}/generated/speed.mp4",
        "status": "completed",
    }
    payload = {
        "id": pid, "name": "Speed Render", "fps": 25, "seed": 42,
        "resolution_w": 320, "resolution_h": 180, "revision": 0,
        "shots": [shot], "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    response = app_client.post(
        f"/projects/{pid}/render",
        json={"settings": {"width": 320, "height": 180, "fps": 25}, "track_states": {}},
    )
    assert response.status_code == 200, response.text
    render_id = response.json()["job_id"]
    status = app_client.get(f"/status/{render_id}").json()
    assert status["status"] == "completed", status

    output = generated / f"{render_id}.mp4"
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(output),
    ]))
    assert float(probe["format"]["duration"]) == pytest.approx(1.0, abs=0.15)


def test_render_api_matches_computed_timeline(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    _make_clip(generated / "red.mp4", "red")
    _make_clip(generated / "blue.mp4", "blue")
    _make_clip(generated / "green.mp4", "green")
    _make_audio(generated / "music.m4a")

    shots = []
    for i, (name, color) in enumerate((("red", "red"), ("blue", "blue"))):
        shots.append({
            "id": name,
            "prompt": name,
            "negative_prompt": "",
            "seed": 42,
            "width": 320,
            "height": 180,
            "fps": 25,
            "num_frames": 25,
            "cfg_scale": 1.0,
            "enhance_prompt": False,
            "upscale": False,
            "pipeline_override": "auto",
            "track_index": 0,
            "start_frame": 0,
            "trim_in": 0,
            "trim_out": 0,
            "timeline": [],
            "video_url": f"/projects/{pid}/generated/{name}.mp4",
            "status": "completed",
            "keyframes": (
                [
                    {"frame": 0, "opacity": 0.2, "scale": 1.0, "x": 0, "y": 0},
                    {"frame": 24, "opacity": 1.0, "scale": 1.1, "x": 0.05, "y": 0},
                ]
                if name == "red" else []
            ),
        })
    shots.append({
        "id": "overlay",
        "prompt": "green overlay",
        "negative_prompt": "",
        "seed": 42,
        "width": 320,
        "height": 180,
        "fps": 25,
        "num_frames": 25,
        "cfg_scale": 1.0,
        "enhance_prompt": False,
        "upscale": False,
        "pipeline_override": "auto",
        "track_index": 1,
        "start_frame": 25,
        "trim_in": 0,
        "trim_out": 0,
        "timeline": [],
        "video_url": f"/projects/{pid}/generated/green.mp4",
        "status": "completed",
        "keyframes": [{"frame": 0, "opacity": 0.0}, {"frame": 24, "opacity": 1.0}],
    })
    shots.append({
        "id": "music",
        "prompt": "music",
        "negative_prompt": "",
        "seed": 42,
        "width": 320,
        "height": 180,
        "fps": 25,
        "num_frames": 50,
        "cfg_scale": 1.0,
        "enhance_prompt": False,
        "upscale": False,
        "pipeline_override": "auto",
        "track_index": 2,
        "start_frame": 0,
        "trim_in": 0,
        "trim_out": 0,
        "timeline": [],
        "video_url": f"/projects/{pid}/generated/music.m4a",
        "status": "completed",
    })
    payload = {
        "id": pid,
        "name": "Render",
        "fps": 25,
        "seed": 42,
        "resolution_w": 320,
        "resolution_h": 180,
        "revision": 0,
        "shots": shots,
        "scenes": [],
        "captions": [
            {"id": "cue-1", "start_frame": 0, "end_frame": 25, "text": "Captioned cut"},
        ],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    response = app_client.post(
        f"/projects/{pid}/render",
        json={"settings": {"width": 320, "height": 180, "fps": 25}, "track_states": {}},
    )
    assert response.status_code == 200, response.text
    render_id = response.json()["job_id"]

    status = app_client.get(f"/status/{render_id}")
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["status"] == "completed", body

    output = generated / f"{render_id}.mp4"
    assert output.exists()
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type", "-of", "json", str(output),
    ])
    data = json.loads(probe)
    assert float(data["format"]["duration"]) == pytest.approx(2.0, abs=0.15)
    assert {s["codec_type"] for s in data["streams"]} == {"video", "audio"}
