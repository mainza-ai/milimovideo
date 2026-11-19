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


def test_relink_migrates_job_take_and_lut_references(app_client, project):
    import io
    import os
    from datetime import datetime

    from PIL import Image
    from sqlmodel import Session

    from database import Job, engine

    pid = project["id"]
    old = _upload(app_client, pid, "old.png")
    old_created = next(
        row["created_at"] for row in app_client.get("/uploads", params={"project_id": pid}).json()
        if row["url"] == old["url"]
    )

    state = {
        "id": pid, "name": "Relink Provenance", "fps": 25, "seed": 42, "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "relink-take-shot", "project_id": pid, "index": 0, "sequence_index": 0,
            "prompt": "take", "seed": 1, "width": 32, "height": 24, "num_frames": 1,
            "track_index": 0, "timeline": [], "video_url": old["url"], "status": "completed",
            "last_job_id": "relink-take-job", "selected_take_id": "relink-take-job",
        }],
        "render_lut": old["url"],
    }
    assert app_client.put(f"/projects/{pid}", json=state).status_code == 200

    with Session(engine) as session:
        session.add(Job(
            id="relink-take-job",
            project_id=pid,
            type="shot_generation",
            status="completed",
            output_path=old["url"],
            shot_id="relink-take-shot",
            params_json='{"seed": 1}',
            created_at=datetime.utcnow(),
        ))
        session.commit()

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (90, 20, 20)).save(buffer, format="PNG")
    buffer.seek(0)
    replaced = app_client.post(
        f"/assets/{old['asset_id']}/replace",
        files={"file": ("new.png", buffer, "image/png")},
    )
    assert replaced.status_code == 200, replaced.text
    data = replaced.json()
    assert data["old_url"] == old["url"]
    assert data["old_removed"] is False
    assert data["created_at"] == old_created

    shot = app_client.get(f"/projects/{pid}").json()
    assert shot["shots"][0]["video_url"] == data["url"]
    assert shot["render_lut"] == data["url"]

    takes = app_client.get("/shots/relink-take-shot/takes").json()["takes"]
    assert takes[0]["url"] == data["url"]
    assert takes[0]["missing"] is False

    # Type changes under an existing take are refused.
    buffer = io.BytesIO(b"not really a video")
    buffer.seek(0)
    refused = app_client.post(
        f"/assets/{old['asset_id']}/replace",
        files={"file": ("new.mp4", buffer, "video/mp4")},
    )
    assert refused.status_code == 409


def test_take_replace_relinks_in_place_and_can_purge_old(app_client, project):
    import io
    import os
    from datetime import datetime

    from PIL import Image
    from sqlmodel import Session

    from database import Job, engine

    pid = project["id"]
    old = _upload(app_client, pid, "take-old.png")
    old_path = os.path.join(
        os.environ["MILIMO_PROJECTS_DIR"], pid, "assets", "general", old["filename"],
    )

    with Session(engine) as session:
        session.add(Job(
            id="take-replace-job",
            project_id=pid,
            type="shot_generation",
            status="completed",
            output_path=old["url"],
            shot_id="take-replace-shot",
            params_json='{"seed": 2}',
            created_at=datetime.utcnow(),
        ))
        session.commit()

    state = {
        "id": pid, "name": "Take Relink", "fps": 25, "seed": 42, "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "take-replace-shot", "project_id": pid, "index": 0, "sequence_index": 0,
            "prompt": "take", "seed": 2, "width": 32, "height": 24, "num_frames": 1,
            "track_index": 0, "timeline": [], "video_url": old["url"], "status": "completed",
            "last_job_id": "take-replace-job", "selected_take_id": "take-replace-job",
        }],
    }
    assert app_client.put(f"/projects/{pid}", json=state).status_code == 200

    buffer = io.BytesIO()
    Image.new("RGB", (44, 33), (10, 120, 10)).save(buffer, format="PNG")
    buffer.seek(0)
    replaced = app_client.post(
        "/shots/take-replace-shot/takes/take-replace-job/replace",
        files={"file": ("take-new.png", buffer, "image/png")},
        data={"purge_old": "true"},
    )
    assert replaced.status_code == 200, replaced.text
    data = replaced.json()
    assert data["missing"] is False
    assert data["old_removed"] is True
    assert not os.path.exists(old_path)

    shot = app_client.get(f"/projects/{pid}").json()["shots"][0]
    assert shot["video_url"] == data["url"]
    takes = app_client.get("/shots/take-replace-shot/takes").json()["takes"]
    assert takes[0]["url"] == data["url"]
