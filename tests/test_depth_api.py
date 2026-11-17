def test_depth_requires_the_local_model(app_client, project, monkeypatch):
    from services import model_registry as registry_module

    monkeypatch.setattr(
        registry_module.model_registry,
        "get_model_info",
        lambda model_id: None,
    )
    response = app_client.post(f"/projects/{project['id']}/depth", json={
        "image_path": f"/projects/{project['id']}/assets/general/frame.jpg",
        "shot_id": None,
    })
    assert response.status_code == 409
    assert "not downloaded" in response.json()["detail"]
