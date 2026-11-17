from pathlib import Path

from sqlmodel import Session

import config
from database import Job, engine


def test_delete_asset_removes_a_job_output_directory(app_client, project):
    pid = project["id"]
    target = Path(config.PROJECTS_DIR) / pid / "generated" / "delete_me"
    target.mkdir(parents=True, exist_ok=True)
    (target / "frame.jpg").write_bytes(b"x")

    with Session(engine) as session:
        session.add(Job(
            id="job_delete_dir",
            project_id=pid,
            type="image",
            status="completed",
            output_path=f"/projects/{pid}/generated/delete_me",
        ))
        session.commit()

    response = app_client.delete("/assets/job_delete_dir")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "deleted"
    assert not target.exists()

    with Session(engine) as session:
        job = session.get(Job, "job_delete_dir")
        assert job is not None
        assert job.status == "deleted"
        assert job.output_path is None


def test_uploads_derive_a_thumbnail_url_for_generated_video(app_client, project):
    pid = project["id"]
    video = Path(config.PROJECTS_DIR) / pid / "generated" / "clip.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"x")

    with Session(engine) as session:
        session.add(Job(
            id="job_thumb_fallback",
            project_id=pid,
            type="video",
            status="completed",
            output_path=f"/projects/{pid}/generated/clip.mp4",
        ))
        session.commit()

    response = app_client.get("/uploads")
    assert response.status_code == 200, response.text
    entry = next(item for item in response.json() if item["id"] == "job_thumb_fallback")
    assert entry["thumbnail"] == f"/projects/{pid}/thumbnails/clip_thumb.jpg"
