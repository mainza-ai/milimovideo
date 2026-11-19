"""GET → PUT round-trips must not send read-only timestamps back to SQLite."""

from __future__ import annotations


def test_project_get_put_round_trip(app_client):
    project = app_client.post("/projects", json={"name": "Round Trip"}).json()
    pid = project["id"]

    state = {
        **project,
        "shots": [{
            "id": "rt-shot",
            "project_id": pid,
            "index": 0,
            "sequence_index": 0,
            "prompt": "round trip",
            "seed": 7,
            "width": 768,
            "height": 512,
            "num_frames": 25,
            "track_index": 0,
            "timeline": [],
            "status": "completed",
            "video_url": "/projects/rt-shot.mp4",
        }],
        "scenes": [{"id": "rt-scene", "index": 0, "name": "Scene", "script_content": "INT. ROOM"}],
        "captions": [{"id": "rt-cue", "start_frame": 0, "end_frame": 10, "text": "Hello"}],
        "titles": [{
            "id": "rt-title", "text": "Title", "start_frame": 0, "end_frame": 10,
            "x": 0.5, "y": 0.5, "size": 48, "color": "#ffffff", "box": True,
        }],
        "render_lut": None,
    }
    created = app_client.put(f"/projects/{pid}", json=state)
    assert created.status_code == 200, created.text

    read = app_client.get(f"/projects/{pid}").json()
    assert read["shots"][0]["created_at"]
    created_at = read["shots"][0]["created_at"]

    # The exact body a client reads back must be writable.
    round_trip = app_client.put(f"/projects/{pid}", json=read)
    assert round_trip.status_code == 200, round_trip.text

    again = app_client.get(f"/projects/{pid}").json()
    assert again["shots"][0]["created_at"] == created_at
    assert again["shots"][0]["prompt"] == "round trip"
    assert again["captions"][0]["text"] == "Hello"
    assert again["titles"][0]["text"] == "Title"
    assert again["scenes"][0]["name"] == "Scene"
