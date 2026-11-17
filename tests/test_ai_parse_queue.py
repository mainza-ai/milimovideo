def test_ai_parse_is_queued_not_run_in_request(app_client, project, monkeypatch):
    import job_utils

    called = {}

    async def fake_enqueue(job_id, kind, params, **extra):
        called.update({"job_id": job_id, "kind": kind, "params": params})
        return "queue-id"

    monkeypatch.setattr(job_utils, "enqueue_gpu_job", fake_enqueue)

    response = app_client.post(f"/projects/{project['id']}/storyboard/ai-parse-jobs", json={
        "script_text": "INT. STUDIO - NIGHT\n\nA filmmaker edits.",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert called["kind"] == "ai_parse"
    assert called["params"]["project_id"] == project["id"]
