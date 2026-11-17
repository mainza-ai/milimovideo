import logging
import requests
import io
import os
from PIL import Image
import config
from models.flux_wrapper import flux_inpainter
from database import Job, get_session
import uuid

logger = logging.getLogger("inpainting_manager")

class InpaintingManager:
    def __init__(self):
        self.sam_url = f"http://localhost:{config.SAM_SERVICE_PORT}"

    async def get_mask_from_sam(self, image_path: str, points: list) -> str:
        """
        Calls SAM microservice to get a mask.
        Returns path to saved mask.
        """
        if not os.path.exists(image_path):
             logger.error(f"Image not found: {image_path}")
             return None

        # Prepare file
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}  # Field name matches server: image
                # points should be list of lists
                data = {"points": str(points), "multimask": "false"} 
                
                logger.info(f"Requesting mask from SAM at {self.sam_url} with points {points}")
                res = requests.post(f"{self.sam_url}/predict/mask", files=files, data=data, timeout=120)
                
                if res.status_code == 200:
                    # Response is a PNG image
                    mask_content = res.content
                    
                    # Save mask to same dir as image but with suffix
                    dir_name = os.path.dirname(image_path)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    mask_filename = f"{base_name}_mask_{uuid.uuid4().hex[:6]}.png"
                    mask_path = os.path.join(dir_name, mask_filename)
                    
                    with open(mask_path, "wb") as f_out:
                         f_out.write(mask_content)
                         
                    logger.info(f"Mask saved to {mask_path}")
                    return mask_path
                else:
                    logger.error(f"SAM Service failed: {res.text}")
                    return None
        except Exception as e:
            logger.error(f"SAM Service connection failed: {e}")
            return None

    async def get_mask_from_text(self, image_path: str, text_prompt: str, confidence: float = 0.5) -> str:
        """
        Calls SAM microservice to get a mask using a text prompt.
        Uses Sam3Processor's text-prompted segmentation.
        Returns path to saved mask.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        try:
            with open(image_path, "rb") as f:
                files = {"image": ("image.jpg", f, "image/jpeg")}
                data = {"text": text_prompt, "confidence": str(confidence)}

                logger.info(f"Requesting text-based mask from SAM: '{text_prompt}'")
                res = requests.post(
                    f"{self.sam_url}/segment/text", files=files, data=data, timeout=120
                )

                if res.status_code == 200:
                    # Response is a PNG mask
                    dir_name = os.path.dirname(image_path)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    mask_filename = f"{base_name}_textmask_{uuid.uuid4().hex[:6]}.png"
                    mask_path = os.path.join(dir_name, mask_filename)

                    with open(mask_path, "wb") as f_out:
                        f_out.write(res.content)

                    logger.info(f"Text-based mask saved to {mask_path}")
                    return mask_path
                else:
                    logger.error(f"SAM text segmentation failed: {res.text}")
                    return None
        except Exception as e:
            logger.error(f"SAM text segmentation connection failed: {e}")
            return None

    async def detect_objects(self, image_path: str, text_prompt: str, confidence: float = 0.5) -> dict:
        """
        Calls SAM /detect endpoint for multi-object detection.
        Returns the full detection results (masks, boxes, scores).
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {"objects": [], "count": 0}

        try:
            with open(image_path, "rb") as f:
                files = {"image": ("image.jpg", f, "image/jpeg")}
                data = {"text": text_prompt, "confidence": str(confidence)}

                res = requests.post(f"{self.sam_url}/detect", files=files, data=data, timeout=120)

                if res.status_code == 200:
                    return res.json()
                else:
                    logger.error(f"SAM detection failed: {res.text}")
                    return {"objects": [], "count": 0}
        except Exception as e:
            logger.error(f"SAM detection connection failed: {e}")
            return {"objects": [], "count": 0}

    async def process_inpaint(self, job_id: str, image_path: str, mask_path: str, prompt: str):
        """
        Runs Flux In-Painting on the image using the mask.
        """
        from job_utils import active_jobs, update_job_progress
        from events import event_manager
        import job_events
        from memory_manager import memory_manager

        decision = memory_manager.admit_job(job_id, "image", {})
        if not decision.ok:
            from services.memory.models import MemoryDeferred

            if decision.retryable:
                raise MemoryDeferred(decision.reason or "waiting for memory")
            await job_events.terminal(
                job_id, "failed", stage="admission", error=decision.reason,
                message=f"Inpaint blocked: {decision.reason}",
                data={"downgrade": decision.downgrade},
            )
            return

        try:
            logger.info(f"Starting Inpaint Job {job_id} with prompt: {prompt}")
            
            # Mark as processing
            if job_id in active_jobs:
                active_jobs[job_id]["status"] = "processing"
                active_jobs[job_id]["status_message"] = "Loading Flux…"
                active_jobs[job_id]["stage"] = "loading"
            update_job_progress(job_id, 5, "Loading Flux…", stage="loading")

            if not os.path.exists(image_path):
                 raise FileNotFoundError(f"Source image not found: {image_path}")
            if not os.path.exists(mask_path):
                 raise FileNotFoundError(f"Mask not found: {mask_path}")

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Step callback for progress updates
            def step_callback(step, total):
                if total > 0:
                    pct = 10 + int((step / total) * 80)
                    update_job_progress(
                        job_id, pct, f"In-painting ({step}/{total})",
                        stage="inpainting", step=step, total=total,
                    )
                if active_jobs.get(job_id, {}).get("cancelled"):
                    raise RuntimeError(f"Job {job_id} cancelled by user.")

            # Flux is blocking; never run it on the event loop.
            import asyncio
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: flux_inpainter.inpaint(
                    image, mask, prompt,
                    guidance=2.0, enable_ae=True, enable_true_cfg=False,
                    step_callback=step_callback,
                ),
            )
            
            # Save result
            out_dir = os.path.dirname(image_path)
            if "assets" in out_dir:
                 out_dir = out_dir.replace("assets", "generated")
            
            os.makedirs(out_dir, exist_ok=True)
            
            out_filename = f"inpaint_{job_id}.jpg"
            out_path = os.path.join(out_dir, out_filename)
            
            result.save(out_path)
            logger.info(f"In-painting completed: {out_path}")
            
            # Build web-accessible URL from absolute path (platform-safe)
            import config
            rel = os.path.relpath(out_path, config.PROJECTS_DIR)
            relative_url = "/projects/" + rel.replace(os.sep, "/")

            # Create Asset record so inpainted image appears in project gallery
            # and can be dragged to timeline conditioning for video re-generation
            asset_id = None
            try:
                import json
                from database import Asset, engine as db_engine
                from sqlmodel import Session
                from datetime import datetime
                
                # Extract project_id from path: .../projects/{project_id}/generated/...
                parts = out_path.replace(config.PROJECTS_DIR, "").strip(os.sep).split(os.sep)
                project_id = parts[0] if parts else None
                
                if project_id:
                    img = Image.open(out_path)
                    meta = {
                        "prompt": prompt,
                        "source": "inpaint",
                        "original_image": image_path,
                        "mask": mask_path
                    }
                    with Session(db_engine) as session:
                        new_asset = Asset(
                            project_id=project_id,
                            type="image",
                            path=out_path,
                            url=relative_url,
                            filename=out_filename,
                            width=img.width,
                            height=img.height,
                            meta_json=json.dumps(meta),
                            created_at=datetime.utcnow()
                        )
                        session.add(new_asset)
                        session.commit()
                        asset_id = new_asset.id
                        logger.info(f"Created Asset record {asset_id} for inpainted image")
            except Exception as e_asset:
                logger.error(f"Failed to create Asset record for inpaint: {e_asset}")

            await job_events.terminal(
                job_id, "completed", stage="completed", message="Inpaint complete",
                output_path=relative_url, thumbnail_path=relative_url, actual_frames=1, progress=100,
                data={"type": "inpaint", "asset_id": asset_id},
            )

            # Cleanup active_jobs
            if job_id in active_jobs:
                active_jobs[job_id]["status"] = "completed"
                active_jobs[job_id]["progress"] = 100
                active_jobs[job_id]["status_message"] = "Complete"
            
            return out_path
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            
            if "cancelled" in str(e).lower():
                await job_events.terminal(
                    job_id, "cancelled", stage="cancelled", message="Cancelled"
                )
            else:
                await job_events.terminal(
                    job_id, "failed", stage="inpainting", error=str(e),
                    message=f"Inpaint failed: {e}",
                )

            # Cleanup active_jobs
            if job_id in active_jobs:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["status_message"] = f"Failed: {e}"
            
            raise e

inpainting_manager = InpaintingManager()
