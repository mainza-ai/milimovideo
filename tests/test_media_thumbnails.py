import shutil
import subprocess
from pathlib import Path

import pytest

import config
from sqlmodel import Session

from database import Asset, engine


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_uploaded_video_gets_a_real_poster(app_client, project):
    pid = project["id"]
    source = Path(config.PROJECTS_DIR) / pid / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=teal:s=160x90:r=25:d=1",
        "-pix_fmt", "yuv420p", str(source),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with source.open("rb") as handle:
        response = app_client.post(
            "/upload",
            files={"file": ("uploaded-clip.mp4", handle, "video/mp4")},
            data={"project_id": pid, "type": "general"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "video"
    assert body["thumbnail"].startswith(f"/projects/{pid}/thumbnails/")

    thumb_path = Path(config.PROJECTS_DIR) / pid / "thumbnails" / Path(body["thumbnail"]).name
    assert thumb_path.exists()
    assert thumb_path.stat().st_size > 0

    listed = app_client.get("/uploads").json()
    entry = next(item for item in listed if item["id"] == body["asset_id"])
    assert entry["thumbnail"] == body["thumbnail"]

    with Session(engine) as session:
        asset = session.get(Asset, body["asset_id"])
        assert asset is not None
        assert asset.thumbnail_url == body["thumbnail"]


def test_uploaded_video_gets_a_720p_review_proxy(app_client, project):
    pid = project["id"]
    source = Path(config.PROJECTS_DIR) / pid / "proxy-source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:r=25:d=1",
        "-pix_fmt", "yuv420p", str(source),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with source.open("rb") as handle:
        response = app_client.post(
            "/upload",
            files={"file": ("proxy-clip.mp4", handle, "video/mp4")},
            data={"project_id": pid, "type": "general"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["proxy"].startswith(f"/projects/{pid}/proxies/")

    proxy_path = Path(config.PROJECTS_DIR) / pid / "proxies" / Path(body["proxy"]).name
    assert proxy_path.exists()
    assert proxy_path.stat().st_size > 0

    listed = app_client.get("/uploads", params={"project_id": pid}).json()
    entry = next(item for item in listed if item["id"] == body["asset_id"])
    assert entry["proxy"] == body["proxy"]
