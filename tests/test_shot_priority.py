"""Per-shot generation steps and queue priority persist and round-trip."""

from __future__ import annotations


def test_shot_priority_and_steps_round_trip(app_client, project):
    pid = project["id"]
    state = {
        "id": pid,
        "name": "Priority Film",
        "fps": 25,
        "seed": 42,
        "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "priority-shot",
            "project_id": pid,
            "index": 0,
            "sequence_index": 0,
            "prompt": "urgent shot",
            "seed": 1,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "num_inference_steps": 8,
            "priority": 3,
            "track_index": 0,
            "timeline": [],
        }],
    }
    saved = app_client.put(f"/projects/{pid}", json=state)
    assert saved.status_code == 200, saved.text

    loaded = app_client.get(f"/projects/{pid}").json()
    shot = loaded["shots"][0]
    assert shot["num_inference_steps"] == 8
    assert shot["priority"] == 3
