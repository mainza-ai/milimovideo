def _shot(shot_id, prompt="a rainy street"):
    return {
        "id": shot_id,
        "prompt": prompt,
        "negative_prompt": "blurry",
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
        "shot_type": "medium",
        "timeline": [],
    }


def _project_with_shot(app_client, project):
    pid = project["id"]
    shot_id = f"plan-shot-{pid}"
    payload = {
        "id": pid,
        "name": "Planner",
        "fps": 25,
        "seed": 42,
        "resolution_w": 768,
        "resolution_h": 512,
        "revision": 0,
        "shots": [_shot(shot_id)],
        "scenes": [],
    }
    assert app_client.put(f"/projects/{pid}", json=payload).status_code == 200
    return pid, shot_id


def test_direct_plan_is_a_pure_diff(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)
    response = app_client.post(f"/projects/{pid}/plan", json={
        "mode": "direct",
        "shot_id": shot_id,
        "text": "seed 7, frames 81, type close-up, prompt: a neon alley at night",
    })
    assert response.status_code == 200, response.text
    plan = response.json()
    kinds = {step["kind"]: step for step in plan["steps"]}
    assert kinds["set_seed"]["before"] == 42
    assert kinds["set_seed"]["after"] == 7
    assert kinds["set_frames"]["after"] == 81
    assert kinds["set_shot_type"]["after"] == "close_up"
    assert kinds["set_prompt"]["after"] == "a neon alley at night"
    assert plan["estimate"] is None

    # A plan never mutates the document.
    stored = app_client.get(f"/projects/{pid}").json()
    shot = next(item for item in stored["shots"] if item["id"] == shot_id)
    assert shot["seed"] == 42
    assert shot["num_frames"] == 121


def test_generate_plan_has_estimate_and_no_job(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)
    response = app_client.post(f"/projects/{pid}/plan", json={
        "mode": "generate",
        "shot_id": shot_id,
        "text": "the same street, now sunrise",
    })
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["steps"][0]["kind"] == "generate"
    assert plan["estimate"]["gpu_seconds"] > 0
    assert plan["estimate"]["vram_mb"] > 0


def test_ask_and_find_work_without_a_shot(app_client, project):
    pid = project["id"]
    ask = app_client.post(f"/projects/{pid}/plan", json={"mode": "ask", "text": "how long is the cut?"})
    assert ask.status_code == 200, ask.text
    assert ask.json()["steps"][0]["kind"] == "answer"

    find = app_client.post(f"/projects/{pid}/plan", json={"mode": "find", "text": "neon"})
    assert find.status_code == 200, find.text
    assert find.json()["steps"][0]["kind"] == "search"
    assert find.json()["steps"][0]["after"] == "neon"


def test_direct_requires_a_shot_and_reports_unknown_directives(app_client, project):
    pid, shot_id = _project_with_shot(app_client, project)
    missing = app_client.post(f"/projects/{pid}/plan", json={"mode": "direct", "text": "seed 9"})
    assert missing.status_code == 400

    vague = app_client.post(f"/projects/{pid}/plan", json={
        "mode": "direct", "shot_id": shot_id, "text": "improve the vibe",
    })
    assert vague.status_code == 200, vague.text
    assert vague.json()["steps"] == []
    assert vague.json()["warnings"]
