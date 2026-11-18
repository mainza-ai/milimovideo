"""Queued Wan 2.2 TI2V-5B generation (low-VRAM video lane)."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

import config
import job_events
from job_utils import (
    active_jobs,
    apply_downgrade,
    update_job_db,
    update_job_progress,
    update_shot_db,
)
from services.memory.models import MemoryDeferred

logger = logging.getLogger("tasks.wan")


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


async def generate_wan_video_task(job_id: str, params: dict) -> None:
    from memory_manager import memory_manager

    project_id = params.get("project_id")
    shot_id = params.get("shot_id") or params.get("id")
    prompt = params.get("prompt") or ""
    negative = params.get("negative_prompt") or ""
    width = int(params.get("width") or 480)
    height = int(params.get("height") or 480)
    num_frames = int(params.get("num_frames") or 49)
    steps = max(1, int(params.get("num_inference_steps") or 20))
    guidance = float(params.get("cfg_scale") or 5.0)
    fps = int(params.get("fps") or 25)
    seed = params.get("seed")
    engine_params = {**params, "engine": "wan"}

    active_jobs[job_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "status_message": "Starting Wan 2.2…",
        "stage": "starting",
        "project_id": project_id,
        "shot_id": shot_id,
        "type": "video",
    }
    update_job_db(job_id, "processing", stage="starting", status_message="Starting Wan 2.2…")

    decision = memory_manager.admit_job(job_id, "video", engine_params)
    if not decision.ok:
        if decision.retryable:
            raise MemoryDeferred(decision.reason or "waiting for memory")
        if decision.downgrade and params.get("auto_downgrade", True):
            before = decision.estimate.total_gb if decision.estimate else 0.0
            label = decision.downgrade.get("label") or "a smaller preset"
            params = apply_downgrade(job_id, params, decision.downgrade)
            width = int(params.get("width") or width)
            height = int(params.get("height") or height)
            num_frames = int(params.get("num_frames") or num_frames)
            if shot_id:
                update_shot_db(shot_id, width=width, height=height, num_frames=num_frames)
            await job_events.retrying(
                job_id, attempt=1,
                message=f"Memory: using {label} for Wan 2.2 (was ~{before:.1f} GB)",
                data={"downgrade": decision.downgrade},
            )
            engine_params = {**params, "engine": "wan"}
            decision = memory_manager.admit_job(job_id, "video", engine_params)
        if not decision.ok:
            message = decision.reason or "Not enough memory for Wan 2.2."
            await job_events.terminal(
                job_id, "failed", stage="admission", error=message,
                message=f"Wan 2.2 blocked: {message}",
                data={"suggestions": decision.suggestions},
            )
            return

    from models.wan_wrapper import WanEngine
    from services.memory import costs

    device = _device()
    wan_cost = costs.WAN_BASE_GB_CUDA if device == "cuda" else costs.WAN_BASE_GB
    memory_manager.prepare_for(
        "video", model_id="wan2.2-5b", cost_gb=wan_cost, job_id=job_id
    )
    engine = WanEngine(config.WAN_MODEL_PATH, device)
    try:
        update_job_progress(job_id, 3, "Loading Wan 2.2…", stage="loading")
        await asyncio.to_thread(engine.load)
        update_job_progress(job_id, 10, "Starting generation…", stage="denoising")

        def cancel_check():
            if active_jobs.get(job_id, {}).get("cancelled"):
                raise RuntimeError(f"Job {job_id} cancelled by user.")

        def on_step(step: int, total: int):
            cancel_check()
            pct = 10 + int((step / max(1, total)) * 80)
            update_job_progress(
                job_id, pct, f"Generating with Wan ({step}/{total})",
                stage="denoising", step=step, total=total,
            )

        frames, actual_frames = await asyncio.to_thread(
            engine.generate,
            prompt,
            negative,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance=guidance,
            seed=seed,
            callback=on_step,
        )

        from file_utils import get_project_output_paths

        paths = get_project_output_paths(job_id, project_id)
        os.makedirs(paths["output_dir"], exist_ok=True)
        os.makedirs(paths["thumbnail_dir"], exist_ok=True)
        output_path = paths["output_path"]
        update_job_progress(job_id, 92, "Encoding video…", stage="encoding")
        await asyncio.to_thread(engine.export_frames, frames, output_path, fps)

        thumbnail_web = None
        try:
            thumb_path = paths["thumbnail_path"]
            await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-i", output_path, "-ss", "00:00:00.500",
                 "-vframes", "1", thumb_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            thumbnail_web = (
                f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
            )
        except Exception as exc:
            logger.warning("Wan thumbnail failed: %s", exc)

        output_url = f"/projects/{project_id}/generated/{os.path.basename(output_path)}"
        if shot_id:
            update_shot_db(
                shot_id,
                video_url=output_url,
                thumbnail_url=thumbnail_web,
                last_job_id=job_id,
                selected_take_id=job_id,
                status="completed",
            )
        update_job_progress(job_id, 100, "Done", stage="completed")
        await job_events.terminal(
            job_id, "completed", stage="completed", message="Done",
            output_path=output_url, thumbnail_path=thumbnail_web,
            actual_frames=actual_frames, progress=100,
            data={"type": "video", "engine": "wan"},
        )
    except Exception as exc:
        if "cancelled" in str(exc).lower():
            await job_events.terminal(
                job_id, "cancelled", stage="cancelled", message="Cancelled"
            )
        else:
            logger.exception("Wan job %s failed: %s", job_id, exc)
            await job_events.terminal(
                job_id, "failed", stage="denoising", error=str(exc),
                message=f"Wan generation failed: {exc}",
            )
    finally:
        engine.unload()
        active_jobs.pop(job_id, None)
