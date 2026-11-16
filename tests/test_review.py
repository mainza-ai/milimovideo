def _project_with_shot(app_client, project):
    pid = project["id"]
    shot_id = f"review-shot-{pid}"
    payload = {
        "id": pid,
        "name": "Review",
        "fps": 25,
        "seed": 42,
        "resolution_w": 768,
        "resolution_h": 512,
        "revision": 0,
        "shots": [{
            "id": shot_id,
            "prompt": "a rainy street",
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
        }],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    return pid, shot_id


def test_frame_comment_lifecycle(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)

    created = app_client.post(f"/projects/{pid}/review/comments", json={
        "shot_id": shot_id, "frame": 42, "body": "The rain reads as static here.",
    })
    assert created.status_code == 200, created.text
    comment = created.json()
    assert comment["frame"] == 42
    assert comment["status"] == "open"

    listed = app_client.get(f"/projects/{pid}/review").json()
    assert listed["summary"]["comments"] == 1
    assert listed["summary"]["open"] == 1
    assert listed["comments"][0]["body"].startswith("The rain")

    resolved = app_client.patch(
        f"/projects/{pid}/review/comments/{comment['id']}",
        json={"status": "resolved"},
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["status"] == "resolved"

    deleted = app_client.delete(f"/projects/{pid}/review/comments/{comment['id']}")
    assert deleted.status_code == 200
    assert app_client.get(f"/projects/{pid}/review").json()["comments"] == []


def test_comment_validation_rejects_blank_and_foreign_shots(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)
    blank = app_client.post(f"/projects/{pid}/review/comments", json={
        "shot_id": shot_id, "body": "   ",
    })
    assert blank.status_code == 400
    foreign = app_client.post(f"/projects/{pid}/review/comments", json={
        "shot_id": "does-not-exist", "body": "x",
    })
    assert foreign.status_code == 404


def test_approval_is_upserted_per_shot(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)

    first = app_client.post(f"/projects/{pid}/review/approvals", json={
        "shot_id": shot_id, "status": "approved", "note": "ship it",
    })
    assert first.status_code == 200, first.text

    second = app_client.post(f"/projects/{pid}/review/approvals", json={
        "shot_id": shot_id, "status": "changes_requested", "note": "rain is too loud",
    })
    assert second.status_code == 200, second.text

    review = app_client.get(f"/projects/{pid}/review").json()
    assert len(review["approvals"]) == 1
    assert review["approvals"][0]["status"] == "changes_requested"
    assert review["approvals"][0]["body"] == "rain is too loud"
    assert review["summary"]["approved"] == 0
    assert review["summary"]["changes_requested"] == 1
