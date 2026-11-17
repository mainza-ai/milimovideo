def test_integrity_reports_missing_media_and_orphans(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid, "name": "Integrity", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [{
            "id": "integrity-shot", "prompt": "x", "negative_prompt": "", "seed": 42,
            "width": 768, "height": 512, "fps": 25, "num_frames": 25,
            "cfg_scale": 2, "enhance_prompt": True, "upscale": False,
            "pipeline_override": "auto", "track_index": 0, "start_frame": 0,
            "trim_in": 0, "trim_out": 0, "timeline": [],
            "video_url": f"/projects/{pid}/generated/missing.mp4",
            "thumbnail_url": f"/projects/{pid}/thumbnails/missing.jpg",
            "status": "completed",
        }],
        "scenes": [{"id": "empty-scene", "name": "Empty", "index": 0}],
        "captions": [{"id": "bad", "start_frame": 10, "end_frame": 5, "text": "x"}],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    # Simulate legacy bad data that bypassed the save sanitizer.
    from sqlmodel import Session
    from database import Project, engine

    with Session(engine) as session:
        stored = session.get(Project, pid)
        stored.captions = '[{"id":"bad","start_frame":10,"end_frame":5,"text":"x"}]'
        session.add(stored)
        session.commit()

    response = app_client.get(f"/projects/{pid}/integrity")
    assert response.status_code == 200, response.text
    body = response.json()
    kinds = {issue["kind"] for issue in body["issues"]}
    assert "missing_video" in kinds
    assert "missing_thumbnail" in kinds
    assert "empty_scene" in kinds
    assert "invalid_caption" in kinds
    assert body["ok"] is False


def test_repair_clears_missing_media_and_orphans(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid, "name": "Repair", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [{
            "id": "repair-shot", "prompt": "x", "negative_prompt": "", "seed": 42,
            "width": 768, "height": 512, "fps": 25, "num_frames": 25,
            "cfg_scale": 2, "enhance_prompt": True, "upscale": False,
            "pipeline_override": "auto", "track_index": 0, "start_frame": 0,
            "trim_in": 0, "trim_out": 0, "timeline": [],
            "video_url": f"/projects/{pid}/generated/missing.mp4",
            "thumbnail_url": f"/projects/{pid}/thumbnails/missing.jpg",
        }],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    board = app_client.put(f"/projects/{pid}/board", json={
        "nodes": [{"id": "orphan-node", "x": 0, "y": 0}],
        "areas": [],
        "links": [{"id": "orphan-link", "source": "orphan-node", "target": "repair-shot", "kind": "thread"}],
        "revision": 0,
    })
    assert board.status_code == 200, board.text

    from sqlmodel import Session
    from database import Project, engine

    with Session(engine) as session:
        stored = session.get(Project, pid)
        stored.captions = '[{"id":"bad","start_frame":10,"end_frame":5,"text":"x"}]'
        session.add(stored)
        session.commit()

    repaired = app_client.post(f"/projects/{pid}/repair")
    assert repaired.status_code == 200, repaired.text
    counts = repaired.json()["counts"]
    assert counts["missing_video"] == 1
    assert counts["missing_thumbnail"] == 1
    assert counts["orphan_board_nodes"] == 1
    assert counts["orphan_board_links"] == 1
    assert counts["captions_sanitized"] == 1

    body = app_client.get(f"/projects/{pid}").json()
    assert body["shots"][0]["video_url"] is None
    assert body["shots"][0]["thumbnail_url"] is None
    assert body["captions"][0]["end_frame"] > body["captions"][0]["start_frame"]
    layout = app_client.get(f"/projects/{pid}/board").json()["layout"]
    assert layout["nodes"] == []
    assert layout["links"] == []

    assert app_client.get(f"/projects/{pid}/integrity").json()["ok"] is True
