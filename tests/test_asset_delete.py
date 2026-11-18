"""Deleting a Library asset removes the file, poster, and shot pointer."""

from __future__ import annotations

import io
import os

from PIL import Image


def test_delete_asset_clears_the_file_and_shot_reference(app_client, project):
    pid = project["id"]
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (20, 40, 60)).save(buffer, format="PNG")
    buffer.seek(0)
    upload = app_client.post(
        "/upload",
        files={"file": ("delete-me.png", buffer, "image/png")},
        data={"project_id": pid},
    )
    assert upload.status_code == 200, upload.text
    asset = upload.json()
    uploaded_path = os.path.join(
        os.environ["MILIMO_PROJECTS_DIR"], pid, "assets", "general", asset["filename"],
    )
    assert os.path.exists(uploaded_path)

    state = {
        "id": pid, "name": "Delete Test", "fps": 25, "seed": 42, "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "delete-shot", "project_id": pid, "index": 0, "sequence_index": 0,
            "prompt": "asset", "seed": 1, "width": 64, "height": 48, "num_frames": 1,
            "track_index": 0, "timeline": [], "video_url": asset["url"],
            "thumbnail_url": asset["url"], "status": "completed",
        }],
    }
    assert app_client.put(f"/projects/{pid}", json=state).status_code == 200

    deleted = app_client.delete(f"/assets/{asset['asset_id']}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["status"] == "deleted"

    shot = app_client.get(f"/projects/{pid}").json()["shots"][0]
    assert shot["video_url"] is None
    assert shot["thumbnail_url"] is None
    uploads = app_client.get("/uploads").json()
    assert all(row.get("url") != asset["url"] for row in uploads)
    assert not os.path.exists(uploaded_path)
