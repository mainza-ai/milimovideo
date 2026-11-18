"""OTIO export: the computed cut must survive an NLE handoff."""

from __future__ import annotations

import json
import os

import opentimelineio as otio
import pytest


def _shot(project_id: str, shot_id: str, index: int, *, track=0, frames=100,
          trim_in=0, trim_out=0, speed=1.0, start=0):
    return {
        "id": shot_id,
        "project_id": project_id,
        "index": index,
        "sequence_index": index,
        "prompt": f"shot {index}",
        "seed": 42,
        "width": 512,
        "height": 320,
        "num_frames": frames,
        "fps": 25,
        "trim_in": trim_in,
        "trim_out": trim_out,
        "speed": speed,
        "track_index": track,
        "start_frame": start,
        "video_url": f"/projects/{project_id}/generated/{shot_id}.mp4",
        "timeline": [],
    }


def test_otio_export_keeps_v1_v2_a1_and_trims(app_client, project):
    pid = project["id"]
    generated = os.path.join(os.environ["MILIMO_PROJECTS_DIR"], pid, "generated")
    os.makedirs(generated, exist_ok=True)
    for shot_id in ("v1-a", "v1-b", "v2-a", "a1-a"):
        with open(os.path.join(generated, f"{shot_id}.mp4"), "wb") as handle:
            handle.write(b"placeholder")

    state = {
        "id": pid,
        "name": "OTIO Film",
        "fps": 25,
        "seed": 42,
        "resolution_w": 512,
        "resolution_h": 320,
        "revision": project.get("revision", 0),
        "scenes": [],
        "shots": [
            _shot(pid, "v1-a", 0, frames=100),
            _shot(pid, "v1-b", 1, frames=80, trim_in=10, trim_out=20, speed=2.0),
            _shot(pid, "v2-a", 2, track=1, frames=50, start=120),
            _shot(pid, "a1-a", 3, track=2, frames=60, start=40),
        ],
    }
    saved = app_client.put(f"/projects/{pid}", json=state)
    assert saved.status_code == 200, saved.text

    response = app_client.get(f"/projects/{pid}/otio")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment" in response.headers["content-disposition"]

    timeline = otio.adapters.read_from_string(response.text, "otio_json")
    assert timeline.name == "OTIO Film"
    tracks = timeline.tracks
    assert [track.name for track in tracks] == ["V1", "V2", "A1"]
    assert [track.kind for track in tracks] == [
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Audio,
    ]

    v1 = [item for item in tracks[0] if isinstance(item, otio.schema.Clip)]
    assert [clip.name for clip in v1] == ["shot 0", "shot 1"]
    assert v1[0].source_range.start_time.value == 0
    assert v1[0].source_range.duration.value == 100
    # 80 total - 10 in - 20 out => 50 source frames at 2x => 25 timeline frames
    assert v1[1].source_range.start_time.value == 10
    assert v1[1].source_range.duration.value == 50
    assert v1[1].metadata["milimo"]["speed"] == 2.0
    effects = [effect for effect in v1[1].effects if effect.effect_name == "LinearTimeWarp"]
    assert effects and effects[0].time_scalar == 2.0

    v2 = [item for item in tracks[1] if isinstance(item, otio.schema.Clip)]
    assert len(v2) == 1
    assert v2[0].metadata["milimo"]["track_index"] == 1
    # A gap of 120 frames precedes the V2 clip.
    assert isinstance(tracks[1][0], otio.schema.Gap)
    assert tracks[1][0].duration().value == 120

    a1 = [item for item in tracks[2] if isinstance(item, otio.schema.Clip)]
    assert len(a1) == 1
    assert isinstance(tracks[2][0], otio.schema.Gap)
    assert tracks[2][0].duration().value == 40

    assert timeline.metadata["milimo"]["duration_frames"] == 170


def test_otio_export_uses_missing_reference_for_offline_media(app_client, project):
    pid = project["id"]
    state = {
        "id": pid,
        "name": "Offline Film",
        "fps": 25,
        "seed": 42,
        "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [_shot(pid, "offline", 0, frames=50)],
    }
    app_client.put(f"/projects/{pid}", json=state)
    response = app_client.get(f"/projects/{pid}/otio")
    timeline = otio.adapters.read_from_string(response.text, "otio_json")
    clip = timeline.tracks[0][0]
    assert isinstance(clip.media_reference, otio.schema.MissingReference)
    assert clip.media_reference.available_range.duration.value == 50
