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
