def _shot(shot_id, prompt):
    return {
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


def test_storyboard_commit_archives_before_replacing(app_client, project):
    pid = project["id"]
    payload = {
        "id": pid, "name": "Archive", "fps": 25, "seed": 42,
        "resolution_w": 768, "resolution_h": 512, "revision": 0,
        "shots": [_shot("archive-old", "old shot")],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200

    commit = app_client.post(f"/projects/{pid}/storyboard/commit", json={
        "scenes": [{
            "name": "Scene One",
            "content": "new",
            "shots": [{"action": "new shot", "shot_type": "medium"}],
        }],
        "script_text": "INT. NEW - DAY",
    })
    assert commit.status_code == 200, commit.text

    archive = app_client.get(f"/projects/{pid}/storyboard/archive").json()["archive"]
    assert archive is not None
    assert archive["shots"][0]["id"] == "archive-old"
    assert archive["saved_at"]
    loaded = app_client.get(f"/projects/{pid}").json()
    assert all(shot["id"] != "archive-old" for shot in loaded["shots"])
