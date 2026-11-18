"""Queued LTX-2.5 generation (gated diffusers engine)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone

import config
import job_events
from job_utils import (
    active_jobs,
    update_job_db,
    update_job_progress,
    update_shot_db,
)
from services.memory.models import MemoryDeferred

logger = logging.getLogger("tasks.ltx25")


async def generate_ltx25_video_task(job_id: str, params: dict) -> None:
    from memory_manager import memory_manager
    from models.ltx25_wrapper import LTX25Engine, device_name
    from services.memory import costs

    project_id = params.get("project_id")
    shot_id = params.get("shot_id") or params.get("id")
    prompt = params.get("prompt") or ""
    negative = params.get("negative_prompt") or ""
    width = int(params.get("width") or 768)
    height = int(params.get("height") or 512)
    num_frames = int(params.get("num_frames") or 121)
    steps = max(1, int(params.get("num_inference_steps") or 40))
    guidance = float(params.get("cfg_scale") or 4.0)
    fps = int(params.get("fps") or 25)
    seed = params.get("seed")

    active_jobs[job_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "status_message": "Starting LTX-2.5…",
        "stage": "starting",
        "project_id": project_id,
        "shot_id": shot_id,
        "type": "video",
    }
    update_job_db(job_id, "processing", stage="starting", status_message="Starting LTX-2.5…")

    estimate = costs.estimate("video", {**params, "engine": "ltx25"})
    decision = memory_manager.admit_job(job_id, "video", {
        **params, "engine": "ltx25",
    })
    if not decision.ok:
        raise MemoryDeferred(decision.reason or "Not enough memory for LTX-2.5")
    memory_manager.prepare_for(
        "video",
        model_id="ltx25",
        cost_gb=estimate.base_gb,
        job_id=job_id,
        kind="video",
        allow_admission=False,
    )

    engine = LTX25Engine(config.LTX25_MODEL_PATH, device_name())
    from file_utils import get_project_output_paths

    paths = get_project_output_paths(job_id, project_id)
    output_path = paths["output_path"]
    thumbnail_path = paths["thumbnail_path"]
    os.makedirs(paths["output_dir"], exist_ok=True)
    os.makedirs(paths["thumbnail_dir"], exist_ok=True)

    started = asyncio.get_running_loop().time()
    try:
        update_job_progress(job_id, 2, "Loading LTX-2.5…", stage="loading")
        await asyncio.to_thread(engine.load)

        def on_step(step: int, total: int) -> None:
            if active_jobs.get(job_id, {}).get("cancelled"):
                raise RuntimeError("cancelled")
            pct = 8 + int(84 * step / max(1, total))
            update_job_progress(
                job_id, pct, f"LTX-2.5 step {step}/{total}",
                stage="denoising", step=step, total=total,
            )

        update_job_progress(job_id, 6, "Generating LTX-2.5 video…", stage="denoising")
        frames, audio, actual_frames = await asyncio.to_thread(
            engine.generate,
            prompt,
            negative,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=steps,
            guidance=guidance,
            seed=seed,
            callback=on_step,
        )

        update_job_progress(job_id, 94, "Encoding LTX-2.5 video…", stage="encoding")
        await asyncio.to_thread(engine.export_frames, frames, output_path, fps)

        audio_path = os.path.join(paths["output_dir"], f"{job_id}_audio.wav")
        has_audio = False
        if audio is not None:
            has_audio = await asyncio.to_thread(engine.export_audio, audio, audio_path)
        if has_audio and os.path.exists(audio_path):
            muxed = output_path.replace(".mp4", "_muxed.mp4")
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ffmpeg", "-y", "-i", output_path, "-i", audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", muxed,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os.replace(muxed, output_path)
            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        thumbnail_web = None
        for offset in ("00:00:00.500", "00:00:00"):
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["ffmpeg", "-y", "-ss", offset, "-i", output_path,
                     "-vframes", "1", thumbnail_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if os.path.exists(thumbnail_path) and os.path.getsize(thumbnail_path) > 0:
                    thumbnail_web = (
                        f"/projects/{project_id}/thumbnails/{os.path.basename(thumbnail_path)}"
                    )
                    break
            except Exception:
                continue
        if not thumbnail_web:
            logger.warning("LTX-2.5 thumbnail failed for %s", output_path)

        output_url = f"/projects/{project_id}/generated/{os.path.basename(output_path)}"
        from database import Asset, Session, engine as db_engine
        asset_id = uuid.uuid4().hex
        with Session(db_engine) as session:
            session.add(Asset(
                id=asset_id,
                project_id=project_id,
                type="video",
                path=output_path,
                url=output_url,
                filename=os.path.basename(output_path),
                thumbnail_url=thumbnail_web,
                width=width,
                height=height,
                duration=actual_frames / max(1, fps),
                created_at=datetime.now(timezone.utc),
                meta_json=json.dumps({
                    "engine": "ltx25",
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "seed": seed,
                    "steps": steps,
                    "fps": fps,
                    "frames": actual_frames,
                    "audio": has_audio,
                }),
            ))
            session.commit()

        if shot_id:
            update_shot_db(
                shot_id,
                video_url=output_url,
                thumbnail_url=thumbnail_web,
                last_job_id=job_id,
                selected_take_id=job_id,
                status="completed",
            )

        elapsed = round(asyncio.get_running_loop().time() - started, 2)
        update_job_progress(job_id, 100, "LTX-2.5 complete", stage="completed")
        await job_events.terminal(
            job_id,
            "completed",
            stage="completed",
            message="LTX-2.5 video ready",
            output_path=output_url,
            thumbnail_path=thumbnail_web,
            actual_frames=actual_frames,
            progress=100,
            data={
                "type": "video",
                "engine": "ltx25",
                "asset_id": asset_id,
                "audio": has_audio,
                "elapsed_seconds": elapsed,
            },
        )
    except Exception as exc:
        if "cancelled" in str(exc).lower():
            await job_events.terminal(
                job_id, "cancelled", stage="cancelled", message="LTX-2.5 cancelled"
            )
        else:
            logger.exception("LTX-2.5 job %s failed: %s", job_id, exc)
            await job_events.terminal(
                job_id,
                "failed",
                stage="denoising",
                error=str(exc),
                message=f"LTX-2.5 failed: {exc}",
            )
    finally:
        engine.unload()
        active_jobs.pop(job_id, None)
