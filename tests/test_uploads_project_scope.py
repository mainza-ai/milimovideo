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

    buffer = io.BytesIO(b"not a real video")
    buffer.seek(0)
    kind_change = app_client.post(
        "/shots/take-replace-shot/takes/take-replace-job/replace",
        files={"file": ("kind.mp4", buffer, "video/mp4")},
    )
    assert kind_change.status_code == 409

    buffer = io.BytesIO(b"not a real video")
    buffer.seek(0)
    kind_change_allowed = app_client.post(
        "/shots/take-replace-shot/takes/take-replace-job/replace",
        files={"file": ("kind.mp4", buffer, "video/mp4")},
        data={"allow_type_change": "true"},
    )
    assert kind_change_allowed.status_code == 200, kind_change_allowed.text
    assert kind_change_allowed.json()["type_changed"] is True


def test_relink_legacy_empty_path_uses_project_assets(app_client, project):
    import io
    import os

    from PIL import Image
    from sqlmodel import Session

    from database import Asset, engine

    pid = project["id"]
    uploaded = _upload(app_client, pid, "legacy.png")
    with Session(engine) as session:
        asset = session.get(Asset, uploaded["asset_id"])
        asset.path = ""
        session.add(asset)
        session.commit()

    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), (20, 20, 20)).save(buffer, format="PNG")
    buffer.seek(0)
    replaced = app_client.post(
        f"/assets/{uploaded['asset_id']}/replace",
        files={"file": ("legacy-new.png", buffer, "image/png")},
    )
    assert replaced.status_code == 200, replaced.text
    data = replaced.json()
    assert data["url"].startswith(f"/projects/{pid}/assets/general/")
    assert os.path.exists(data["path"])


def test_relink_purge_removes_old_thumbnail_and_respects_shared_references(app_client, project):
    import io
    import os

    from sqlmodel import Session

    from database import Asset, engine
    from PIL import Image

    pid = project["id"]
    uploaded = _upload(app_client, pid, "thumbs.png")
    thumbs_dir = os.path.join(os.environ["MILIMO_PROJECTS_DIR"], pid, "thumbnails")
    os.makedirs(thumbs_dir, exist_ok=True)
    old_thumb_path = os.path.join(thumbs_dir, "old_thumb.jpg")
    Image.new("RGB", (16, 16), (50, 50, 50)).save(old_thumb_path, format="JPEG")
    with Session(engine) as session:
        asset = session.get(Asset, uploaded["asset_id"])
        asset.thumbnail_url = f"/projects/{pid}/thumbnails/old_thumb.jpg"
        session.add(asset)
        session.commit()

    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), (20, 20, 20)).save(buffer, format="PNG")
    buffer.seek(0)
    replaced = app_client.post(
        f"/assets/{uploaded['asset_id']}/replace",
        files={"file": ("thumbs-new.png", buffer, "image/png")},
        data={"purge_old": "true"},
    )
    assert replaced.status_code == 200, replaced.text
    data = replaced.json()
    assert data["old_removed"] is True
    assert not os.path.exists(old_thumb_path)

    # A duplicate Asset row pinning the same path blocks purge.
    other = _upload(app_client, pid, "shared.png")
    with Session(engine) as session:
        source = session.get(Asset, other["asset_id"])
        session.add(Asset(
            id="shared-copy", project_id=pid, type="image",
            path=source.path, url=source.url, filename=source.filename,
        ))
        session.commit()
    buffer = io.BytesIO()
    Image.new("RGB", (18, 18), (5, 5, 5)).save(buffer, format="PNG")
    buffer.seek(0)
    shared = app_client.post(
        f"/assets/{other['asset_id']}/replace",
        files={"file": ("shared-new.png", buffer, "image/png")},
        data={"purge_old": "true"},
    )
    assert shared.status_code == 200, shared.text
    assert shared.json()["old_shared"] is True
    assert shared.json()["old_removed"] is False


def test_repair_clears_take_pointers_for_missing_video(app_client, project):
    pid = project["id"]
    state = {
        "id": pid, "name": "Repair Lineage", "fps": 25, "seed": 42, "scenes": [],
        "revision": project.get("revision", 0),
        "shots": [{
            "id": "f9-repair-shot", "project_id": pid, "index": 0, "sequence_index": 0,
            "prompt": "missing", "seed": 1, "width": 32, "height": 24, "num_frames": 1,
            "track_index": 0, "timeline": [], "status": "completed",
            "video_url": f"/projects/{pid}/generated/missing.mp4",
            "last_job_id": "missing-job", "selected_take_id": "missing-job",
            "favorite_take_id": "missing-job",
        }],
    }
    assert app_client.put(f"/projects/{pid}", json=state).status_code == 200
    repair = app_client.post(f"/projects/{pid}/repair")
    assert repair.status_code == 200, repair.text
    shot = app_client.get(f"/projects/{pid}").json()["shots"][0]
    assert shot["video_url"] is None
    assert shot["selected_take_id"] is None
    assert shot["favorite_take_id"] is None
    assert shot["last_job_id"] is None
    assert shot["status"] == "pending"
    assert repair.json()["counts"]["take_pointer_cleared"] == 1


def test_shot_created_at_is_read_only_in_schema(app_client):
    schema = app_client.get("/openapi.json").json()
    created_at = schema["components"]["schemas"]["ShotState"]["properties"]["created_at"]
    assert created_at.get("readOnly") is True
