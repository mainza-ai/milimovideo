def _shot(shot_id, prompt, **overrides):
    data = {
        "id": shot_id,
        "prompt": prompt,
        "negative_prompt": "",
        "seed": 42,
        "width": 768,
        "height": 512,
        "fps": 25,
        "num_frames": 121,
        "cfg_scale": 2.0,
        "enhance_prompt": True,
        "upscale": False,
        "pipeline_override": "auto",
        "track_index": 0,
        "start_frame": 0,
        "trim_in": 0,
        "trim_out": 0,
        "timeline": [],
    }
    data.update(overrides)
    return data


def test_project_roundtrip_order_and_conditioning(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid,
        "name": "Roundtrip",
        "fps": 25,
        "seed": 42,
        "resolution_w": 768,
        "resolution_h": 512,
        "revision": 0,
        "shots": [
            _shot(
                "a",
                "first",
                sequence_index=0,
                height=704,
                fps=30,
                shot_type="close_up",
                timeline=[
                    {"type": "image", "path": "/projects/x/a.png", "frame_index": 25, "strength": 0.8}
                ],
            ),
            _shot("b", "second", sequence_index=1),
        ],
        "scenes": [],
    }
    response = app_client.put(f"/projects/{pid}", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1

    response = app_client.get(f"/projects/{pid}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["id"] for s in body["shots"]] == ["a", "b"]
    first = body["shots"][0]
    assert first["height"] == 704
    assert first["fps"] == 30
    assert first["shot_type"] == "close_up"
    assert first["timeline"][0]["frame_index"] == 25
    assert first["timeline"][0]["strength"] == 0.8
    assert body["revision"] == 1


def test_stale_revision_returns_conflict(app_client, project):
    pid = project["id"]
    base = {
        "id": pid,
        "name": "Conflict",
        "fps": 25,
        "seed": 42,
        "resolution_w": 768,
        "resolution_h": 512,
        "revision": 0,
        "shots": [],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=base).status_code == 200
    assert app_client.put(f"/projects/{pid}", json=base).status_code == 409


def test_take_history_and_selection(app_client, project):
    from datetime import datetime
    from sqlmodel import Session
    from database import Job, engine

    pid = project["id"]
    shot = _shot("take-shot", "take test")
    payload = {
        "id": pid, "name": "Takes", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [shot], "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    with Session(engine) as session:
        session.add(Job(
            id="job_take_a", project_id=pid, shot_id="take-shot", type="shot_generation",
            status="completed", progress=100, output_path=f"/projects/{pid}/generated/a.mp4",
            thumbnail_path=f"/projects/{pid}/thumbnails/a.jpg", created_at=datetime.utcnow(),
            params_json='{"seed": 7, "num_frames": 25}',
        ))
        session.add(Job(
            id="job_take_b", project_id=pid, shot_id="take-shot", type="shot_generation",
            status="completed", progress=100, output_path=f"/projects/{pid}/generated/b.mp4",
            thumbnail_path=f"/projects/{pid}/thumbnails/b.jpg", created_at=datetime.utcnow(),
            params_json='{"seed": 8, "num_frames": 25}',
        ))
        session.commit()

    response = app_client.get("/shots/take-shot/takes")
    assert response.status_code == 200, response.text
    takes = response.json()["takes"]
    assert len(takes) == 2
    assert sum(1 for take in takes if take["selected"]) <= 1

    select = app_client.post("/shots/take-shot/select-take", json={"job_id": "job_take_a"})
    assert select.status_code == 200, select.text
    selected = app_client.get("/shots/take-shot/takes").json()["takes"]
    assert any(take["job_id"] == "job_take_a" and take["selected"] for take in selected)


def test_project_takes_lists_completed_outputs(app_client, project):
    from datetime import datetime
    from sqlmodel import Session
    from database import Job, engine

    pid = project["id"]
    shot = _shot("takes-shot", "take library", sequence_index=0)
    payload = {
        "id": pid, "name": "Take Library", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [shot], "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    with Session(engine) as session:
        session.add(Job(
            id="library_take_1", project_id=pid, shot_id="takes-shot",
            type="shot_generation", status="completed", progress=100,
            output_path=f"/projects/{pid}/generated/one.mp4",
            created_at=datetime.utcnow(), params_json='{"seed": 3}',
        ))
        session.add(Job(
            id="library_take_2", project_id=pid, shot_id="takes-shot",
            type="advanced", status="failed", progress=20,
            output_path=None, created_at=datetime.utcnow(),
        ))
        session.commit()

    response = app_client.get(f"/projects/{pid}/takes")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["takes"][0]["job_id"] == "library_take_1"
    assert body["takes"][0]["shot_label"] == "take library"


def test_shot_speed_roundtrip(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid, "name": "Speed", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [_shot("speed-shot", "fast", sequence_index=0, speed=2.0)],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    loaded = app_client.get(f"/projects/{pid}").json()
    assert loaded["shots"][0]["speed"] == 2.0


def test_project_script_content_roundtrip(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid, "name": "Script", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [], "scenes": [],
        "script_content": "INT. STUDIO - NIGHT\n\nA filmmaker edits the cut.",
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    loaded = app_client.get(f"/projects/{pid}").json()
    assert loaded["script_content"].startswith("INT. STUDIO")


def test_captions_roundtrip_sorted_and_sanitized(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid, "name": "Captions", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [], "scenes": [],
        "captions": [
            {"id": "late", "start_frame": 50, "end_frame": 75, "text": "later"},
            {"id": "early", "start_frame": 0, "end_frame": 25, "text": " earlier "},
            {"id": "blank", "start_frame": 10, "end_frame": 20, "text": "   "},
        ],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    captions = app_client.get(f"/projects/{pid}").json()["captions"]
    assert [cue["id"] for cue in captions] == ["early", "late"]
    assert captions[0]["text"] == "earlier"


def test_board_layout_roundtrip_and_revision(app_client, project):
    pid = project["id"]
    layout = {
        "nodes": [
            {"id": "one", "x": 10, "y": 20, "width": 300, "height": 230, "pinned": True},
            {"id": "two", "x": 380, "y": 20, "width": 300, "height": 230, "pinned": False},
        ],
        "areas": [{
            "id": "area_act1", "title": "Act I", "x": 0, "y": -40,
            "width": 720, "height": 360, "color": "#06b2eb",
        }],
        "links": [{"id": "link_1", "source": "one", "target": "two", "kind": "blocking"}],
        "revision": 0,
    }
    response = app_client.put(f"/projects/{pid}/board", json=layout)
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1

    fetched = app_client.get(f"/projects/{pid}/board").json()
    assert fetched["layout"]["nodes"][0]["id"] == "one"
    assert fetched["layout"]["nodes"][0]["pinned"] is True
    assert fetched["layout"]["areas"][0]["title"] == "Act I"
    assert fetched["layout"]["links"][0]["kind"] == "blocking"

    stale = {**layout, "revision": 0}
    assert app_client.put(f"/projects/{pid}/board", json=stale).status_code == 409


def test_split_shot_keeps_order_and_timeline(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid,
        "name": "Split",
        "fps": 25,
        "seed": 42,
        "resolution_w": 768,
        "resolution_h": 512,
        "revision": 0,
        "shots": [
            _shot("one", "one", num_frames=121),
            _shot("two", "two", sequence_index=1),
        ],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    response = app_client.post("/shots/one/split", json={"split_frame": 50})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["new_shot"]["trim_in"] == 50
    assert body["original_shot"]["trim_out"] == 71
    assert body["new_shot"]["sequence_index"] == 1

    reloaded = app_client.get(f"/projects/{pid}").json()["shots"]
    assert [s["id"] for s in reloaded] == ["one", body["new_shot"]["id"], "two"]
