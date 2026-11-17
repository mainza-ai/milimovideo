import json

from sqlmodel import Session

from database import QueuedJob, engine, init_db


def test_active_jobs_includes_queued_position_and_estimate(app_client, project):
    init_db()
    pid = project["id"]
    with Session(engine) as session:
        session.add(QueuedJob(
            job_id="queued-job-1",
            kind="video",
            status="queued",
            params_json=json.dumps({
                "project_id": pid, "width": 768, "height": 512, "num_frames": 121,
            }),
        ))
        session.commit()

    jobs = app_client.get(f"/projects/{pid}/active_jobs").json()
    queued = next(job for job in jobs if job["job_id"] == "queued-job-1")
    assert queued["position"] == 1
    assert queued["estimate_seconds"] > 0
    assert queued["status"] == "queued"
