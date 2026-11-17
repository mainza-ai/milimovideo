"""Local monocular depth estimation for stage overlays."""

import asyncio
import os
import uuid

import config
from database import Job
from events import event_manager
from job_utils import active_jobs, update_job_db, update_job_progress
from file_utils import resolve_path

_depth_pipeline = None


def _load_pipeline(model_path: str):
    global _depth_pipeline
    if _depth_pipeline is not None:
        return _depth_pipeline

    from transformers import pipeline

    device = -1
    try:
        import torch
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = -1

    try:
        _depth_pipeline = pipeline(
            "depth-estimation",
            model=model_path,
            device=device,
        )
    except Exception:
        _depth_pipeline = pipeline(
            "depth-estimation",
            model=model_path,
            device=-1,
        )
    return _depth_pipeline


def _run_depth(image_path: str, model_path: str):
    from PIL import Image

    pipe = _load_pipeline(model_path)
    image = Image.open(image_path).convert("RGB")
    result = pipe(image)
    depth = result["depth"]
    return depth


async def depth_task(job_id: str, params: dict) -> None:
    active_jobs.setdefault(job_id, {})
    active_jobs[job_id].update({
        "status": "processing",
        "progress": 5,
        "status_message": "Loading depth model…",
        "type": "depth",
        "project_id": params.get("project_id"),
        "cancelled": False,
    })

    try:
        from services.model_registry import model_registry

        model = model_registry.get_model_info("depth-anything-v2-small")
        if not model or not os.path.exists(model.absolute_path):
            raise RuntimeError(
                "Depth Anything V2 Small is not downloaded. Download it in Setup."
            )

        image_path = resolve_path(params.get("image_path"))
        if not image_path or not os.path.exists(image_path):
            raise RuntimeError("Depth source image is missing")

        update_job_progress(job_id, 25, "Estimating depth…")
        depth = await asyncio.to_thread(_run_depth, image_path, model.absolute_path)

        project_id = params.get("project_id")
        output_dir = os.path.join(config.PROJECTS_DIR, project_id or "", "generated")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"depth_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(output_dir, filename)
        depth.save(output_path)

        url = f"/projects/{project_id}/generated/{filename}"
        update_job_progress(job_id, 100, "Depth complete")
        update_job_db(job_id, "completed", output_path=url)
        await event_manager.broadcast("depth_complete", {
            "job_id": job_id,
            "project_id": project_id,
            "url": url,
        })
    except Exception as exc:
        update_job_db(job_id, "failed", error=str(exc))
        await event_manager.broadcast("depth_failed", {
            "job_id": job_id, "error": str(exc),
        })
