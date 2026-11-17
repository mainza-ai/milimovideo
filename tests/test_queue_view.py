import json

import pytest
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

    # Leave no undispatchable row behind: it would outrank later tests' jobs.
    with Session(engine) as session:
        row = session.get(QueuedJob, queued["job_id"])
        if row:
            session.delete(row)
            session.commit()


@pytest.mark.asyncio
async def test_spawn_background_tracks_and_releases_tasks():
    import asyncio
    import job_utils

    started = asyncio.Event()

    async def work():
        started.set()

    task = job_utils._spawn_background(work())
    assert task in job_utils._background_tasks
    await started.wait()
    await task
    await asyncio.sleep(0)
    assert task not in job_utils._background_tasks


def test_queue_pause_and_resume_endpoints(app_client):
    paused = app_client.post("/jobs/queue/pause")
    assert paused.status_code == 200
    assert paused.json()["paused"] is True
    assert app_client.get("/jobs/queue").json()["paused"] is True
    assert app_client.get("/health").json()["queue_paused"] is True

    resumed = app_client.post("/jobs/queue/resume")
    assert resumed.status_code == 200
    assert resumed.json()["paused"] is False
    assert app_client.get("/jobs/queue").json()["paused"] is False


@pytest.mark.asyncio
async def test_paused_queue_does_not_dispatch(app_client):
    import asyncio
    from database import QueuedJob
    import job_utils

    with Session(engine) as session:
        session.add(QueuedJob(id="queue-pause-test", job_id="job-pause", kind="unknown", status="queued"))
        session.commit()

    job_utils.set_queue_paused(True)
    task = asyncio.create_task(job_utils.run_queued_job("queue-pause-test"))
    try:
        await asyncio.sleep(0.1)
        with Session(engine) as session:
            assert session.get(QueuedJob, "queue-pause-test").status == "queued"
    finally:
        job_utils.set_queue_paused(False)

    await task
    with Session(engine) as session:
        assert session.get(QueuedJob, "queue-pause-test").status == "failed"
