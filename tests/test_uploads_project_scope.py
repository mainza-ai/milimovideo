"""Library media is scoped to one project; the global view stays opt-in."""

from __future__ import annotations

import io

from PIL import Image


def _upload(app_client, project_id: str, name: str):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 30, 50)).save(buffer, format="PNG")
    buffer.seek(0)
    response = app_client.post(
        "/upload",
        files={"file": (name, buffer, "image/png")},
        data={"project_id": project_id},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_uploads_filters_by_project(app_client, project):
    other = app_client.post("/projects", json={"name": "Other Project"}).json()
    mine = _upload(app_client, project["id"], "mine.png")
    theirs = _upload(app_client, other["id"], "theirs.png")

    scoped = app_client.get("/uploads", params={"project_id": project["id"]})
    assert scoped.status_code == 200
    urls = {row["url"] for row in scoped.json()}
    assert mine["url"] in urls
    assert theirs["url"] not in urls
    assert all("missing" in row for row in scoped.json())

    global_urls = {row["url"] for row in app_client.get("/uploads").json()}
    assert {mine["url"], theirs["url"]} <= global_urls


def test_uploads_marks_missing_files(app_client, project):
    uploaded = _upload(app_client, project["id"], "gone.png")
    import os

    path = os.path.join(
        os.environ["MILIMO_PROJECTS_DIR"], project["id"], "assets", "general", uploaded["filename"],
    )
    os.remove(path)

    rows = app_client.get("/uploads", params={"project_id": project["id"]}).json()
    row = next(item for item in rows if item["url"] == uploaded["url"])
    assert row["missing"] is True


def test_replace_asset_relinks_file_and_shot(app_client, project):
    uploaded = _upload(app_client, project["id"], "old.png")
    import io
    import os
    from PIL import Image

    # Point a shot at the old URL, then replace the file through the relink path.
    state = {
        "id": project["id"], "name": "Relink Test", "fps": 25, "seed": 42, "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "relink-shot", "project_id": project["id"], "index": 0, "sequence_index": 0,
            "prompt": "asset", "seed": 1, "width": 32, "height": 24, "num_frames": 1,
            "track_index": 0, "timeline": [], "video_url": uploaded["url"], "status": "completed",
        }],
    }
    assert app_client.put(f"/projects/{project['id']}", json=state).status_code == 200

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (90, 20, 20)).save(buffer, format="PNG")
    buffer.seek(0)
    replaced = app_client.post(
        f"/assets/{uploaded['asset_id']}/replace",
        files={"file": ("new.png", buffer, "image/png")},
    )
    assert replaced.status_code == 200, replaced.text
    data = replaced.json()
    assert data["id"] == uploaded["asset_id"]
    assert data["url"] != uploaded["url"]
    assert data["missing"] is False
    assert os.path.exists(os.path.join(os.environ["MILIMO_PROJECTS_DIR"], project["id"], "assets", "general", data["filename"]))

    shot = app_client.get(f"/projects/{project['id']}").json()["shots"][0]
    assert shot["video_url"] == data["url"]
