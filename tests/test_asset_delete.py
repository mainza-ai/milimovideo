import shutil
import subprocess
from pathlib import Path

import pytest
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


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_last_frame_extraction_writes_a_jpeg(app_client, project):
    pid = project["id"]
    generated = Path(config.PROJECTS_DIR) / pid / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    video = generated / "last-frame-source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=25:d=1",
        "-pix_fmt", "yuv420p", str(video),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with Session(engine) as session:
        session.add(Job(
            id="job_last_frame",
            project_id=pid,
            type="video",
            status="completed",
            output_path=f"/projects/{pid}/generated/last-frame-source.mp4",
        ))
        session.commit()

    response = app_client.post("/shot/job_last_frame/last-frame")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["url"].endswith("job_last_frame_last.jpg")
    assert Path(body["path"]).exists()
