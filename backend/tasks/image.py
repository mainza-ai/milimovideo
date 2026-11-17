import asyncio
import os
import logging
import uuid
import torch
import json
from datetime import datetime, timezone
from job_utils import update_job_db, active_jobs, update_job_progress, update_shot_db, broadcast_progress
from file_utils import get_base_dir, get_project_output_paths
from events import event_manager
from database import engine, Job, Shot
from sqlmodel import Session
import config
from PIL import Image

logger = logging.getLogger(__name__)

async def generate_image_task(job_id: str, params: dict):
    # Setup paths
    config.setup_paths()
    
    logger.info(f"Starting IMAGE generation for job {job_id}")

    if not config.SKIP_VRAM_CHECK:
        from services.model_loader import model_loader
        can_load, reason = model_loader.can_load_model(config.VRAM_ESTIMATE_IMAGE_GB)
        if not can_load:
            message = f"Insufficient memory for image generation: {reason}"
            logger.error(message)
            update_job_db(job_id, "failed", error=message)
            await event_manager.broadcast("error", {"job_id": job_id, "message": message})
            return

    active_jobs[job_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "eta_seconds": 5,
        "status_message": "Initializing...",
        "project_id": params.get("project_id"),
        "type": "image"
    }
    update_job_db(job_id, "processing")
    
    try:
        prompt = params.get("prompt", "")
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        cfg_scale = params.get("guidance_scale", 2.0)
        project_id = params.get("project_id")
        seed = params.get("seed", 42)
        steps = params.get("num_inference_steps", 25)
        negative_prompt = params.get("negative_prompt", "")
        enable_ae = params.get("enable_ae", True)
        enable_true_cfg = params.get("enable_true_cfg", False)
        
        logger.info(f"DEBUG: Task received guidance_scale={cfg_scale}, negative_prompt='{negative_prompt}', AE={enable_ae}, TrueCFG={enable_true_cfg}")

        # Prompt Enhancement through the `prompt_image` role.
        # Flux has no LTX text encoder, so this role must resolve to an HTTP
        # provider (Ollama, LM Studio, or an OpenAI-compatible server).
        enhance = params.get("enhance_prompt", True)
        if enhance:
            try:
                from llm import enhance_prompt as llm_enhance, role_uses_gemma
                if role_uses_gemma("prompt_image"):
                    logger.warning(
                        "The prompt_image role resolves to Gemma, but image generation uses "
                        "Flux and has no LTX text encoder. Skipping enhancement; assign "
                        "prompt_image to a local HTTP provider in Studio Setup."
                    )
                else:
                    enhanced, _ = llm_enhance(
                        prompt=prompt, is_video=False, seed=seed,
                        cancel_check=lambda: active_jobs.get(job_id, {}).get("cancelled", False),
                    )
                    if enhanced and enhanced != prompt:
                        logger.info(f"Enhanced image prompt: {enhanced[:100]}...")
                        update_job_db(job_id, "processing", enhanced_prompt=enhanced)
                        prompt = enhanced
            except Exception as e:
                logger.warning(f"Image prompt enhancement failed: {e}")
            finally:
                try:
                    from llm import release_local_models
                    release_local_models()
                except Exception:
                    pass
        
        # Flux 2 Logic
        from models.flux_wrapper import flux_inpainter
        
        
        # Merge sources: element_images (legacy/injected) + reference_images (API)
        element_images = params.get("element_images", [])
        if "reference_images" in params and params["reference_images"]:
            element_images.extend(params["reference_images"])
        
        # Resolve 'element_images' or 'reference_images'
        # Resolve Elements / IP-Adapter
        # params['reference_images'] contains element_ids (explicit from UI)
        # params['element_images'] might contain paths from earlier middleware?
        # Let's align with worker.py logic.
        
        from database import Session, engine, Element
        from sqlmodel import select
        from job_utils import resolve_element_image_path
        
        resolved_ip_paths = set() 
        resolved_triggers_injected = []
        
        # 1. explicit IDs
        with Session(engine) as session:
            if element_images:
                # If they are just paths, use them. If IDs, look up.
                for item in element_images:
                    if isinstance(item, str) and not item.startswith("/") and not item.startswith("http"):
                            # Probably an ID
                            el = session.get(Element, item)
                            if el:
                                path = resolve_element_image_path(el.image_path)
                                if path: resolved_ip_paths.add(path)
                                if el.trigger_word and el.trigger_word not in prompt:
                                    resolved_triggers_injected.append(el.trigger_word)
                    else:
                            # It's a path
                            if isinstance(item, str):
                                if item.startswith("/projects"):
                                    # project path
                                    rel = item.removeprefix("/projects/")
                                    resolved_ip_paths.add(os.path.join(get_base_dir(), "projects", rel))
                                else:
                                    resolved_ip_paths.add(item)

            # 2. Scan Prompt for Implicit Triggers (e.g. "@Hero") — cross-project
            all_elements = session.exec(select(Element)).all()
            for el in all_elements:
                if el.trigger_word and el.trigger_word.lower() in prompt.lower():
                    resolved_path = resolve_element_image_path(el.image_path)
                    if resolved_path:
                        logger.info(f"Auto-detected trigger in prompt: {el.trigger_word} (project {el.project_id})")
                        resolved_ip_paths.add(resolved_path)

        element_images = list(resolved_ip_paths)

        # Inject triggers
        if resolved_triggers_injected:
            prompt = f"{prompt} {' '.join(resolved_triggers_injected)}"

        
        loop = asyncio.get_running_loop()
        
        def _run_flux():
             def flux_callback(step, total):
                 if job_id in active_jobs:
                     try:
                         progress_pct = (step / total) * 100
                         msg = f"Generating Image ({step}/{total})"
                         active_jobs[job_id]["progress"] = int(progress_pct)
                         active_jobs[job_id]["status_message"] = msg
                         if active_jobs[job_id].get("cancelled", False):
                             raise RuntimeError(f"Job {job_id} cancelled by user.")
                         
                         # Allow pure cancellation check without clearing status
                         if step >= 0:
                            # Must run in asyncio thread
                            asyncio.run_coroutine_threadsafe(
                                broadcast_progress(job_id, active_jobs[job_id]["progress"], "processing", active_jobs[job_id]["status_message"]),
                                loop
                            )
                     except Exception as e:
                         if "cancelled" in str(e).lower():
                             raise  # Rethrow so outer block catches it
                         logger.error(f"Error in flux callback: {e}")

             img = flux_inpainter.generate_image(
                 prompt=prompt,
                 width=width,
                 height=height,
                 guidance=cfg_scale,
                 num_inference_steps=steps, 
                 ip_adapter_images=element_images, 
                 callback=flux_callback,
                 seed=seed,
                 negative_prompt=negative_prompt,
                 enable_ae=enable_ae,
                 enable_true_cfg=enable_true_cfg
             )
             
             # Save
             if not project_id:
                 raise ValueError("project_id required")
             
             paths = get_project_output_paths(job_id, project_id)
             out_path = paths["output_path"].replace(".mp4", ".jpg")
             thumb_path = paths["thumbnail_path"]
             
             os.makedirs(os.path.dirname(out_path), exist_ok=True)
             os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
             
             img.save(out_path, quality=95)
             img.resize((round(width/4), round(height/4))).save(thumb_path)
             
             web_url = f"/projects/{project_id}/generated/{os.path.basename(out_path)}"
             web_thumb = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
             
             return web_url, web_thumb

        web_url, web_thumb = await loop.run_in_executor(None, _run_flux)
        
        # Reconstruct out_path since it was local to closure
        filename = os.path.basename(web_url)
        out_path = os.path.join(config.PROJECTS_DIR, project_id, "generated", filename)

        # Create Asset Record
        from database import Asset, Session, engine
        asset_id = uuid.uuid4().hex
        
        with Session(engine) as session:
            asset = Asset(
                id=asset_id,
                project_id=project_id,
                type="image",
                path=out_path,
                url=web_url,
                filename=filename,
                width=width,
                height=height,
                created_at=datetime.now(timezone.utc),
                meta_json=json.dumps({
                    "prompt": prompt,
                    "seed": seed,
                    "steps": steps, 
                    "guidance": cfg_scale,
                    "reference_elements": element_images 
                })
            )
            session.add(asset)
            session.commit()
        
        # Update Job
        update_job_db(
            job_id, 
            "completed", 
            output_path=web_url, 
            thumbnail_path=web_thumb,
            actual_frames=1,
            status_message="Flux Image Ready"
        )
        
        # If this is a storyboard thumbnail, write back to the Shot record
        shot_id = params.get("shot_id")
        if params.get("is_thumbnail") and shot_id:
            update_shot_db(shot_id, thumbnail_url=web_url, status="pending")
            logger.info(f"Thumbnail saved to shot {shot_id}: {web_url}")
        
        await event_manager.broadcast("complete", {
            "job_id": job_id, 
            "url": web_url,
            "type": "thumbnail" if params.get("is_thumbnail") else "image",
            "thumbnail_url": web_thumb,
            "asset_id": asset_id,
            "shot_id": shot_id,
        })
        
    except Exception as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Job {job_id} cancelled.")
            update_job_db(job_id, "cancelled")
            await event_manager.broadcast("cancelled", {"job_id": job_id})
        else:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            active_jobs[job_id]["status"] = "failed"
            update_job_db(job_id, "failed", error=str(e))
