import os
import time

import pytest

from services.model_registry import model_registry


def _depth_model_ready() -> bool:
    model = model_registry.get_model_info("depth-anything-v2-small")
    return bool(model and os.path.exists(model.absolute_path))


@pytest.mark.skipif(not _depth_model_ready(), reason="depth model not downloaded")
def test_depth_job_produces_a_map(app_client, project):
    from PIL import Image
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (320, 180), (20, 100, 180)).save(buffer, format="JPEG")
    buffer.seek(0)

    upload = app_client.post(
        "/upload",
        files={"file": ("depth-source.jpg", buffer, "image/jpeg")},
        data={"project_id": project["id"], "type": "reference"},
    )
    assert upload.status_code == 200, upload.text
    image_path = upload.json()["url"]

    response = app_client.post(f"/projects/{project['id']}/depth", json={
        "image_path": image_path,
    })
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    for _ in range(90):
        status = app_client.get(f"/status/{job_id}").json()
        if status["status"] == "completed":
            assert status["video_url"].endswith(".png")
            return
        if status["status"] == "failed":
            pytest.fail(status.get("error") or "depth job failed")
        time.sleep(1)
    pytest.fail("depth job timed out")
