import os
from pathlib import Path

import config
from routes.elements import _parse_points


def test_points_reject_code_execution():
    payload = "__import__('os').system('echo pwned')"
    try:
        _parse_points(payload)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422
    else:
        raise AssertionError("eval payload was accepted")


def test_points_accept_valid_coordinates():
    assert _parse_points("[[10, 20], [30, 40]]") == [[10.0, 20.0], [30.0, 40.0]]


def test_optional_bearer_token_gate(app_client):
    previous = config.API_TOKEN
    config.API_TOKEN = "test-token"
    try:
        assert app_client.get("/projects").status_code == 401
        assert app_client.get(
            "/projects", headers={"Authorization": "Bearer test-token"}
        ).status_code == 200
    finally:
        config.API_TOKEN = previous


def test_upload_traversal_is_sanitized(app_client, project):
    response = app_client.post(
        "/upload",
        files={"file": ("../../evil.png", b"not-an-image", "image/png")},
        data={"project_id": project["id"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "access_path" not in body
    assert "evil.png" in body["filename"]
    assert "/" not in body["filename"] and ".." not in body["filename"]

    escaped = Path(config.PROJECTS_DIR).parent / "evil.png"
    assert not escaped.exists()


def test_upload_rejects_disallowed_extension(app_client, project):
    response = app_client.post(
        "/upload",
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
        data={"project_id": project["id"]},
    )
    assert response.status_code == 415


def test_foreign_origin_is_rejected(app_client):
    response = app_client.get("/projects", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_allowed_origin_gets_cors_header(app_client):
    response = app_client.get("/projects", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_media_range_and_traversal(app_client, project):
    pid = project["id"]
    media_dir = Path(config.PROJECTS_DIR) / pid / "generated"
    media_dir.mkdir(parents=True, exist_ok=True)
    payload = b"0123456789abcdef"
    (media_dir / "clip.mp4").write_bytes(payload)

    full = app_client.get(f"/projects/{pid}/generated/clip.mp4")
    assert full.status_code == 200
    assert full.content == payload

    partial = app_client.get(
        f"/projects/{pid}/generated/clip.mp4",
        headers={"Range": "bytes=4-7"},
    )
    assert partial.status_code == 206
    assert partial.content == b"4567"
    assert partial.headers["content-range"] == "bytes 4-7/16"

    traversal = app_client.get(f"/projects/{pid}/../{pid}_backup/clip.mp4")
    assert traversal.status_code in (403, 404)


def test_upload_rejects_oversize(app_client, project):
    response = app_client.post(
        "/upload",
        files={"file": ("big.png", b"x" * (2 * 1024 * 1024), "image/png")},
        data={"project_id": project["id"]},
    )
    assert response.status_code == 413
