"""Queued AI script parsing.

The parser is heavy (it loads the video pipeline's text encoder), so it must not
run inside the HTTP request or on the event loop. The durable GPU queue owns it.
"""

import asyncio
import json
import os
import uuid

import config
from events import event_manager
from job_utils import update_job_db, update_job_progress, active_jobs


async def ai_parse_task(job_id: str, params: dict) -> None:
    project_id = params.get("project_id")
    script_text = params.get("script_text") or ""
    active_jobs.setdefault(job_id, {})
    active_jobs[job_id].update({
        "status": "processing",
        "progress": 5,
        "status_message": "Loading parser model…",
        "type": "ai_parse",
        "project_id": project_id,
        "cancelled": False,
    })

    try:
        from managers.element_manager import element_manager
        from services.ai_storyboard import ai_parse_script
        from services.script_parser import script_parser

        elements = element_manager.get_elements(project_id)
        parsed_scenes = []
        mode = "fallback"

        try:
            from llm import role_uses_gemma

            text_encoder = None
            if role_uses_gemma("script_parse"):
                from model_engine import manager
                update_job_progress(job_id, 15, "Loading text encoder…")
                pipeline = await manager.load_pipeline("ti2vid")
                text_encoder = pipeline.stage_1_model_ledger.text_encoder()
            update_job_progress(job_id, 35, "Parsing script…")
            parsed_scenes = await asyncio.to_thread(
                ai_parse_script,
                text=script_text,
                text_encoder=text_encoder,
                seed=42,
                project_elements=elements if elements else None,
            )
            del text_encoder
            try:
                from ltx_pipelines.utils.helpers import cleanup_memory
                cleanup_memory()
            except Exception:
                pass
            mode = "ai"
        except Exception as exc:
            update_job_progress(job_id, 35, "AI parse unavailable; using local parser…")
            parsed_scenes = await asyncio.to_thread(
                script_parser.parse_script,
                script_text,
                parse_mode="auto",
                elements=elements if elements else None,
            )
            mode = f"fallback: {exc}"

        workspace = os.path.join(config.PROJECTS_DIR, project_id or "", "workspace")
        os.makedirs(workspace, exist_ok=True)
        filename = f"ai_parse_{uuid.uuid4().hex[:8]}.json"
        path = os.path.join(workspace, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"scenes": parsed_scenes, "mode": mode}, handle)

        url = f"/projects/{project_id}/workspace/{filename}"
        update_job_progress(job_id, 100, "Parse complete")
        update_job_db(job_id, "completed", output_path=url)
        await event_manager.broadcast("ai_parse_complete", {
            "job_id": job_id,
            "project_id": project_id,
            "url": url,
            "mode": mode,
        })
    except Exception as exc:
        update_job_db(job_id, "failed", error=str(exc))
        await event_manager.broadcast("ai_parse_failed", {
            "job_id": job_id, "error": str(exc),
        })
