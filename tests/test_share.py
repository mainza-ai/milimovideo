def _project_with_shot(app_client, project):
    pid = project["id"]
    shot_id = f"share-shot-{pid}"
    payload = {
        "id": pid,
        "name": "Shared Cut",
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


def test_share_link_reads_and_accepts_reviewer_comments(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)

    created = app_client.post(f"/projects/{pid}/share")
    assert created.status_code == 200, created.text
    data = created.json()
    token = data["token"]
    assert data["path"] == f"/studio/share/{token}"
    assert token not in data["share_id"]

    shared = app_client.get(f"/share/{token}")
    assert shared.status_code == 200, shared.text
    body = shared.json()
    assert body["shared"] is True
    assert body["project"]["name"] == "Shared Cut"
    assert body["shots"][0]["id"] == shot_id

    comment = app_client.post(f"/share/{token}/comments", json={
        "shot_id": shot_id, "frame": 42, "body": "Rain is too loud.", "author": "Sam",
    })
    assert comment.status_code == 200, comment.text
    assert comment.json()["author"] == "Sam"

    approval = app_client.post(f"/share/{token}/approvals", json={
        "shot_id": shot_id, "status": "changes_requested", "note": "fix rain",
    })
    assert approval.status_code == 200, approval.text

    updated = app_client.get(f"/share/{token}").json()
    assert updated["summary"]["comments"] == 1
    assert updated["summary"]["changes_requested"] == 1
    assert updated["comments"][0]["frame"] == 42


def test_share_link_is_read_only_and_revocable(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)
    data = app_client.post(f"/projects/{pid}/share").json()
    token = data["token"]

    # The capability cannot mutate the document.
    write = app_client.put(f"/share/{token}/projects/{pid}", json={"name": "hacked"})
    assert write.status_code in (401, 404, 405)

    revoke = app_client.delete(f"/projects/{pid}/share/{data['share_id']}")
    assert revoke.status_code == 200, revoke.text
    assert app_client.get(f"/share/{token}").status_code == 404


def test_invalid_share_token_is_404(app_client):
    assert app_client.get("/share/not-a-real-token").status_code == 404
