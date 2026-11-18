"""Queued ComfyUI workflow execution and output ingestion.

The bridge is generic: a workflow comes in as ComfyUI API-format JSON, runs on
the external ComfyUI server, and every file it writes is copied into the
project workspace as an asset. Shot media is updated when a ``shot_id`` is
supplied.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import config
import job_events
from services.comfy import (
    SAFE_EXTENSION,
    ComfyCancelled,
    ComfyClient,
    ComfyError,
    ComfyUnavailable,
    apply_filename_prefix,
    apply_overrides,
    parse_workflow,
    validate_workflow,
    workflow_size,
)
from services.memory.models import MemoryDeferred
from job_utils import (
    active_jobs,
    update_job_db,
    update_job_progress,
    update_shot_db,
)

logger = logging.getLogger("tasks.comfy")

ASSET_TYPE = {
    "image": "image",
    "video": "video",
    "audio": "audio",
    "text": "text",
}
ASSET_EXTENSIONS = {
    "image": ".png",
    "video": ".mp4",
    "audio": ".wav",
    "text": ".txt",
}


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return cleaned[:40] or "output"


def _output_target(project_id: str, job_id: str, index: int, output) -> dict:
    """Choose a safe project path and media metadata for one output."""
    output_dir = os.path.join(config.PROJECTS_DIR, project_id, "generated")
    os.makedirs(output_dir, exist_ok=True)
    extension = os.path.splitext(output.filename)[1].lower()
    if not SAFE_EXTENSION.match(extension):
        extension = ASSET_EXTENSIONS.get(output.kind, ".bin")
    filename = f"comfy_{job_id}_{index}_{_safe_stem(output.filename)}{extension}"
    path = os.path.join(output_dir, filename)
    url = f"/projects/{project_id}/generated/{filename}"
    return {
        "url": url,
        "path": path,
        "filename": filename,
        "kind": output.kind,
        "type": ASSET_TYPE.get(output.kind, "file"),
        "thumbnail_url": url if output.kind == "image" else None,
        "node_id": output.node_id,
        "source_filename": output.filename,
        "size_bytes": 0,
    }


def _video_thumbnail(project_id: str, record: dict) -> None:
    if record["kind"] != "video":
        return
    try:
        from file_utils import generate_thumbnail

        thumb_path = generate_thumbnail(record["path"])
        if thumb_path:
            record["thumbnail_url"] = (
                f"/projects/{project_id}/generated/{os.path.basename(thumb_path)}"
            )
    except Exception as exc:
        logger.warning("ComfyUI video thumbnail failed: %s", exc)


def _asset_rows(project_id: str, job_id: str, records: list[dict], params: dict, prompt_id: str | None = None) -> list[str]:
    """Insert Asset rows so ComfyUI outputs appear in Library."""
    from database import Asset, Session, engine

    asset_ids: list[str] = []
    for record in records:
        if record["kind"] not in ASSET_TYPE:
            continue
        asset_id = uuid.uuid4().hex
        with Session(engine) as session:
            session.add(Asset(
                id=asset_id,
                project_id=project_id,
                type=ASSET_TYPE[record["kind"]],
                path=record["path"],
                url=record["url"],
                filename=record["filename"],
                thumbnail_url=record["thumbnail_url"],
                created_at=datetime.now(timezone.utc),
                meta_json=json.dumps({
                    "engine": "comfyui",
                    "job_id": job_id,
                    "workflow_node": record["node_id"],
                    "source_filename": record["source_filename"],
                    "prompt": params.get("prompt"),
                    "workflow_id": params.get("workflow_id"),
                    "workflow_name": params.get("workflow_name"),
                    "prompt_id": prompt_id,
                }),
            ))
            session.commit()
        record["asset_id"] = asset_id
        asset_ids.append(asset_id)
    return asset_ids


async def run_comfy_task(job_id: str, params: dict) -> None:
    if not config.COMFY_ENABLED:
        raise ComfyError("ComfyUI bridge is disabled (MILIMO_COMFY_ENABLED=0)")
    project_id = params.get("project_id")
    shot_id = params.get("shot_id")
    if not project_id:
        raise ComfyError("ComfyUI jobs require a project_id")

    workflow = parse_workflow(params.get("workflow"))
    workflow = apply_overrides(workflow, params.get("overrides") or {})
    # Prefix every output with the job id so concurrent ComfyUI runs and manual
    # jobs cannot collide, and so a downloaded file is traceable to its job.
    prefix = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in str(params.get("filename_prefix") or "milimo")
    )[:40] or "milimo"
    workflow = apply_filename_prefix(workflow, f"{prefix}_{job_id}")
    validation = validate_workflow(workflow)
    width, height = workflow_size(workflow)

    active_jobs[job_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "status_message": "Starting ComfyUI…",
        "stage": "starting",
        "project_id": project_id,
        "shot_id": shot_id,
        "type": "comfy",
    }
    update_job_db(job_id, "processing", stage="starting", status_message="Starting ComfyUI…")
    update_job_progress(job_id, 2, "Checking ComfyUI…", stage="starting")

    from memory_manager import memory_manager
    from services.comfy_settings import comfy_settings

    declared_gb = float(comfy_settings.effective_values()["declared_gb"])
    estimate = memory_manager.admit_job(job_id, "comfy", {
        **params,
        "width": width,
        "height": height,
        "estimate_gb": params.get("estimate_gb") or declared_gb,
    })
    if not estimate.ok:
        raise MemoryDeferred(estimate.reason or "Not enough memory for ComfyUI")
    memory_manager.prepare_for(
        "comfy",
        model_id="comfyui:external",
        cost_gb=float(config.COMFY_DECLARED_GB),
        job_id=job_id,
        kind="comfy",
        allow_admission=False,
    )

    client = ComfyClient()
    started = asyncio.get_running_loop().time()
    try:
        health = await asyncio.to_thread(client.health)
        if not health.get("reachable"):
            raise ComfyUnavailable(
                f"ComfyUI is not reachable at {config.COMFY_URL}. "
                "Start Comfy Desktop or set MILIMO_COMFY_URL."
            )
        update_job_progress(
            job_id, 4,
            f"ComfyUI {health.get('version') or ''} ready".strip(),
            stage="starting",
        )

        def progress(payload: dict) -> None:
            update_job_progress(
                job_id,
                int(payload.get("progress") or 0),
                payload.get("message") or "ComfyUI is working…",
                stage=payload.get("stage"),
                step=payload.get("step"),
                total=payload.get("total"),
            )

        prompt_id, entry = await client.run(
            workflow,
            on_progress=progress,
            is_cancelled=lambda: bool(active_jobs.get(job_id, {}).get("cancelled")),
            timeout=float(params.get("timeout_sec") or config.COMFY_JOB_TIMEOUT_SEC),
            estimate_seconds=float(params.get("estimate_seconds") or 30.0),
        )
        outputs = client.collect_outputs(entry)
        if not outputs:
            raise ComfyError(
                "ComfyUI finished but wrote no files. Add a SaveImage (or other "
                "Save) node to the workflow."
            )

        update_job_progress(job_id, 98, "Copying ComfyUI outputs…", stage="saving")
        records: list[dict] = []
        for index, output in enumerate(outputs):
            record = _output_target(project_id, job_id, index, output)
            size = await asyncio.to_thread(client.download_to, output, record["path"])
            if not size:
                logger.warning("ComfyUI output %s was empty", output.filename)
                try:
                    os.remove(record["path"])
                except OSError:
                    pass
                continue
            record["size_bytes"] = size
            _video_thumbnail(project_id, record)
            records.append(record)
        if not records:
            raise ComfyError("ComfyUI outputs could not be downloaded")

        asset_ids = _asset_rows(project_id, job_id, records, params, prompt_id=prompt_id)
        primary = next(
            (record for record in records if record["kind"] in ("video", "image")),
            records[0],
        )
        if shot_id:
            updates = {
                "video_url": primary["url"],
                "thumbnail_url": primary["thumbnail_url"] or primary["url"],
                "last_job_id": job_id,
                "status": "completed",
            }
            if primary["kind"] == "video":
                updates["selected_take_id"] = job_id
            update_shot_db(shot_id, **updates)

        elapsed = round(asyncio.get_running_loop().time() - started, 2)
        public_outputs = [
            {key: value for key, value in record.items() if key != "path"}
            for record in records
        ]
        update_job_progress(job_id, 100, "ComfyUI workflow complete", stage="completed")
        await job_events.terminal(
            job_id,
            "completed",
            stage="completed",
            message="ComfyUI workflow complete",
            output_path=primary["url"],
            thumbnail_path=primary["thumbnail_url"] or primary["url"],
            progress=100,
            data={
                "type": primary["kind"],
                "engine": "comfyui",
                "prompt_id": prompt_id,
                "workflow_id": params.get("workflow_id"),
                "workflow_name": params.get("workflow_name"),
                "project_id": project_id,
                "shot_id": shot_id,
                "asset_id": primary.get("asset_id"),
                "asset_ids": asset_ids,
                "outputs": public_outputs,
                "validation": validation,
                "elapsed_seconds": elapsed,
            },
        )
    except ComfyCancelled as exc:
        await job_events.terminal(
            job_id, "cancelled", stage="cancelled", message=str(exc)
        )
    except MemoryDeferred:
        raise
    except Exception as exc:
        if "cancel" in str(exc).lower():
            await job_events.terminal(
                job_id, "cancelled", stage="cancelled", message="ComfyUI workflow cancelled"
            )
        else:
            logger.exception("ComfyUI job %s failed: %s", job_id, exc)
            await job_events.terminal(
                job_id,
                "failed",
                stage="denoising",
                error=str(exc),
                message=f"ComfyUI failed: {exc}",
            )
    finally:
        client.close()
        active_jobs.pop(job_id, None)
