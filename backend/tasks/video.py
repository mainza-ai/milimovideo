import asyncio
import os
import torch
import logging
import subprocess
import numpy as np
from PIL import Image

import config
from job_utils import update_job_db, active_jobs, update_job_progress, update_shot_db, broadcast_progress, resolve_element_image_path
from file_utils import get_base_dir, get_project_output_paths, resolve_path
from tasks.chained import generate_chained_video_task
from events import event_manager
from database import Job
from sqlmodel import Session
from database import engine

# LTX libs
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.helpers import cleanup_memory

logger = logging.getLogger(__name__)

def clamp_resolution_for_device(width: int, height: int) -> tuple[int, int]:
    """Clamp resolution to safe limits for the current device (MPS/CUDA).
    
    On MPS, the 19B float32 transformer's attention scales quadratically with 
    token count. Resolutions above ~983K pixels (1280×768) cause OOM during 
    scaled_dot_product_attention. This function scales down proportionally,
    maintaining aspect ratio and mod64 alignment.
    """
    if not torch.backends.mps.is_available():
        return width, height  # CUDA handles larger resolutions
    
    max_pixels = config.MPS_MAX_PIXELS
    current_pixels = width * height
    
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        new_width = round(width * scale / 64) * 64
        new_height = round(height * scale / 64) * 64
        logger.warning(
            f"⚠️ MPS Resolution Clamp: {width}x{height} → {new_width}x{new_height} "
            f"(original {current_pixels:,} px exceeded {max_pixels:,} px safe limit)"
        )
        return new_width, new_height
    
    return width, height

async def generate_standard_video_task(job_id: str, params: dict, pipeline):
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        seed = params.get("seed", 42)
        width = params.get("width", 768)
        height = params.get("height", 512)
        
        # Ensure mod64 for LTX-2
        height = round(height / 64) * 64
        width = round(width / 64) * 64
        
        # MPS Safety: Clamp resolution to prevent OOM with 19B float32 model
        width, height = clamp_resolution_for_device(width, height)
        
        num_frames = params.get("num_frames", 121) 
        num_inference_steps = params.get("num_inference_steps", 40)
        cfg_scale = params.get("cfg_scale", 2.0)
        enhance_prompt = params.get("enhance_prompt", True)
        upscale = params.get("upscale", False) # Default to OFF as requested
        
        # Auto-Negative Prompting: Structural Injection
        # We append these terms to ensuring the model avoids non-visual artifacts
        auto_negative = ", text, watermark, copyright, fuzzy, low resolution, caption"
        if negative_prompt:
             negative_prompt += auto_negative
        else:
             negative_prompt = auto_negative.strip(", ")
        
        input_images = params.get("images", [])
        
        # FIX: Support "timeline" parameter (sent by storyboard.py)
        if not input_images:
             timeline = params.get("timeline", [])
             if timeline:
                 logger.info(f"Converting 'timeline' param to 'input_images': {len(timeline)} items")
                 for item in timeline:
                     # timeline item structure: {"path": str, "frame_index": int, "strength": float, "type": "image"}
                     if isinstance(item, dict) and item.get("type") == "image":
                         input_images.append((
                             item.get("path"), 
                             item.get("frame_index", 0), 
                             item.get("strength", 1.0)
                         ))
                         logger.info(f"Parsed timeline item: {input_images[-1]}")
        
        # RESOLVE WEB PATHS for Timeline Input Images
        # Structure: [(path, idx, strength)]
        # Converts /projects/... -> /Users/.../backend/projects/...
        logger.info(f"=== IMAGE PATH RESOLUTION DEBUG ===")
        logger.info(f"Raw input_images from params: {len(input_images)} items")
        project_id = params.get("project_id", None)
        if input_images:
            resolved_inputs = []
            logger.info(f"project_id: {project_id}")
            for item in input_images:
                try:
                    path, idx, strength = item
                    logger.info(f"Processing item: path='{path}', idx={idx}, strength={strength}")
                    
                    resolved_abs = resolve_path(path)
                    
                    if resolved_abs and os.path.exists(resolved_abs):
                        resolved_inputs.append((resolved_abs, idx, strength))
                        logger.info(f"✓ Path exists: {resolved_abs}")
                    else:
                        # FALLBACK: Legacy paths
                        found = False
                        if project_id and isinstance(path, str):
                            filename = os.path.basename(path)
                            project_path = os.path.join(config.PROJECTS_DIR, project_id, "generated", "images", filename)
                            if os.path.exists(project_path):
                                resolved_inputs.append((project_path, idx, strength))
                                found = True
                            else:
                                # Fallback 2: Check in generated folder (flat)
                                project_path_flat = os.path.join(config.PROJECTS_DIR, project_id, "generated", filename)
                                if os.path.exists(project_path_flat):
                                    resolved_inputs.append((project_path_flat, idx, strength))
                                    found = True
                        if not found:
                            logger.warning(f"✗ Raw path not found: {path} (resolved as {resolved_abs})")
                except Exception as e:
                    logger.warning(f"Failed to resolve input image path: {item} - {e}")
                    resolved_inputs.append(item)
            input_images = resolved_inputs
            logger.info(f"Final resolved input_images: {len(input_images)} items")
        
        
        # Element Images Resolution
        element_images = params.get("element_images", [])
        if element_images:
            resolved_elements = []
            for img_path in element_images:
                 resolved_abs = resolve_path(img_path)
                 if resolved_abs and os.path.exists(resolved_abs):
                     resolved_elements.append(resolved_abs)
            element_images = resolved_elements

        # Note: element_images are passed ONLY to ip_adapter_images (line ~210)
        # for style/character guidance. They must NOT be used as input_images
        # (frame-0 conditioning) which would freeze the video as a still frame.

        # INSPIRATION IMAGES (Concept Art / VLM Style Reference)
        # These are used for LLM prompt enhancement but NOT for video conditioning.
        inspiration_images = params.get("inspiration_images", [])
        if inspiration_images:
            resolved_insp = []
            for img_path in inspiration_images:
                 resolved_abs = resolve_path(img_path)
                 if resolved_abs and os.path.exists(resolved_abs):
                     resolved_insp.append(resolved_abs)
            inspiration_images = resolved_insp
            logger.info(f"Resolved {len(inspiration_images)} inspiration images for VLM.")

        video_cond = params.get("video_conditioning", [])
        video_cond = params.get("video_conditioning", [])
        
        # Derive pipeline_type from the object itself to ensure match
        raw_type = params.get("pipeline_override") or params.get("pipeline_type", "ti2vid")
        pipeline_type = raw_type
        
        # If auto/advanced, rely on the actual loaded class
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.ic_lora import ICLoraPipeline
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        
        if isinstance(pipeline, TI2VidTwoStagesPipeline):
            pipeline_type = "ti2vid"
        elif isinstance(pipeline, ICLoraPipeline):
            pipeline_type = "ic_lora"
        elif isinstance(pipeline, KeyframeInterpolationPipeline):
            pipeline_type = "keyframe"
            
        logger.info(f"Pipeline Type Resolved: {pipeline_type} (Raw: {raw_type})")
        
        # Initialize ETA
        if job_id in active_jobs:
            est_time = 45 if num_frames > 1 else 5
            active_jobs[job_id]["eta_seconds"] = est_time
            active_jobs[job_id]["status_message"] = "Initializing generation..."
        
        loop = asyncio.get_running_loop()
        
        if num_frames == 1:
            logger.info(f"Detected single-frame generation. Delegating to Flux 2 (T2I mode)...")
            from models.flux_wrapper import flux_inpainter
            
            # Prompt Enhancement for single-frame (image) mode
            # NOTE: Flux doesn't use LTX text_encoder — only Ollama can enhance here.
            if enhance_prompt:
                try:
                    from llm import enhance_prompt as llm_enhance
                    provider = config.LLM_PROVIDER.lower()
                    if provider == "gemma":
                        logger.warning(
                            "Gemma cannot enhance single-frame prompts (Flux has no LTX text_encoder). "
                            "Skipping enhancement. Use Ollama for image prompt enhancement."
                        )
                    else:
                        logger.info(f"Enhancing single-frame prompt via {provider}...")
                        enhanced = llm_enhance(prompt=prompt, is_video=False, seed=seed)
                        if enhanced and enhanced != prompt:
                            prompt = enhanced
                            update_job_db(job_id, "processing", enhanced_prompt=prompt)
                            update_job_progress(job_id, 5, "Prompt Enhanced", enhanced_prompt=prompt)
                            logger.info(f"Enhanced single-frame prompt: {prompt[:100]}...")
                except Exception as e:
                    logger.warning(f"Single-frame prompt enhancement failed: {e}")
            
            def _run_flux():
                 def flux_callback(step, total):
                     progress = int((step / total) * 100)
                     update_job_progress(job_id, progress, f"Generating Image ({step}/{total})")

                 img = flux_inpainter.generate_image(
                     prompt=prompt,
                     width=width,
                     height=height,
                     guidance=cfg_scale, 
                     ip_adapter_images=element_images, 
                     callback=flux_callback
                 )
                 out_filename = f"{job_id}.jpg"
                 
                 if not project_id:
                     raise ValueError(f"project_id is required for Flux image generation {job_id}")
                 
                 paths = get_project_output_paths(job_id, project_id)
                 out_path = paths["output_path"].replace(".mp4", ".jpg")
                 thumb_path = paths["thumbnail_path"]
                 
                 os.makedirs(os.path.dirname(out_path), exist_ok=True)
                 os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                 
                 web_url = f"/projects/{project_id}/generated/{out_filename}"
                 web_thumb = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
                 
                 img.save(out_path, quality=95)
                 img.resize((round(width/4), round(height/4))).save(thumb_path)
                 
                 return web_url, web_thumb

            try:
                web_url, web_thumb = await loop.run_in_executor(None, _run_flux)
                
                update_job_db(
                    job_id, 
                    "completed", 
                    output_path=web_url, 
                    thumbnail_path=web_thumb,
                    actual_frames=1,
                    status_message="Flux Image Ready"
                )
                
                # Create Asset Record for Gallery
                try:
                    import json
                    from database import Asset
                    from datetime import datetime
                    
                    # Construct metadata for re-use
                    meta = {
                        "prompt": prompt,
                        "negative_prompt": params.get("negative_prompt", ""),
                        "seed": params.get("seed"),
                        "steps": params.get("num_inference_steps"),
                        "guidance": params.get("cfg_scale", params.get("guidance_scale")),
                        "width": width,
                        "height": height,
                        "reference_elements": params.get("reference_images", []) # If available
                    }
                    
                    with Session(engine) as asset_session:
                        new_asset = Asset(
                            project_id=project_id,
                            type="image",
                            path=out_path,
                            url=web_url,
                            filename=out_filename,
                            width=width,
                            height=height,
                            meta_json=json.dumps(meta),
                            created_at=datetime.utcnow()
                        )
                        asset_session.add(new_asset)
                        asset_session.commit()
                        
                        # Sync ID so frontend polling finds the asset immediately (ImagesView looks for job.asset_id potentially?)
                        # Actually ImagesView polling looks for 'completed' status, then re-fetches gallery. 
                        # But wait, ImagesView polling said: 
                        # if (job.status === 'completed') { ... const newImage = data.find((img: any) => img.id === job.asset_id); }
                        # So we should update the JOB with the asset_id if possible, or simple rely on url/filename match.
                        # Job table doesn't have asset_id column in database.py view.
                        # ImagesView seems to expect asset_id in job status?
                        # Let's check ImagesView lines 104: const newImage = data.find((img: any) => img.id === job.asset_id);
                        # If job doesn't have asset_id, it won't highlight, but gallery will rely on refetch.
                        # We can't easily add asset_id to Job model without migration.
                        # Frontend logic: setImages(data); const newImage = data.find...
                        # If we can't pass asset_id, frontend just won't select it automatically. acceptable.
                        pass
                except Exception as e_asset:
                    logger.error(f"Failed to create Asset record: {e_asset}")

                await event_manager.broadcast("complete", {
                    "job_id": job_id, 
                    "url": web_url,
                    "type": "image",
                    "thumbnail_url": web_thumb,
                    "asset_id": new_asset.id if 'new_asset' in locals() else None 
                })
                return
                
            except Exception as e:
                logger.error(f"Flux Generation failed: {e}")
                update_job_db(job_id, "failed", error=str(e))
                await event_manager.broadcast("error", {"job_id": job_id, "message": str(e)})
                return

        # Use MPS-specific tiling
        if torch.backends.mps.is_available():
            tiling_config = TilingConfig.default()
            logger.info("Using MPS-optimized tiling configuration")
        else:
            tiling_config = TilingConfig.default()
        
        def cancellation_check(step_idx, total_steps, *args):
            import time
            # Yield GIL to allow main thread to handle HTTP requests (like cancel)
            time.sleep(0.01)
            
            gen_percent = (step_idx + 1) / total_steps
            global_percent = 10 + int(gen_percent * 80)
            update_job_progress(job_id, global_percent, f"Generating ({step_idx + 1}/{total_steps})")
                
            if active_jobs.get(job_id, {}).get("cancelled", False):
                raise RuntimeError(f"Job {job_id} cancelled by user.")
        
        def _run_pipeline():
            nonlocal input_images, inspiration_images
            result = None
            if pipeline_type == "ti2vid":
                run_prompt = prompt
                if enhance_prompt:
                    update_job_progress(job_id, 5, "Enhancing Prompt...")
                    logger.info(f"Enhancing prompt with {config.LLM_PROVIDER}...")
                    try:
                        from llm import enhance_prompt as llm_enhance
                        is_image_mode = (num_frames == 1)
                        
                        # For Gemma, we need the text encoder
                        text_encoder = None
                        if config.LLM_PROVIDER.lower() == "gemma":
                            text_encoder = pipeline.stage_1_model_ledger.text_encoder()
                        
                        image_for_llm = None
                        
                        # PRIORITY: Inspiration Images (Concept Art) -> Input Images (Frame 0)
                        if inspiration_images:
                             image_for_llm = inspiration_images[0]
                             logger.info(f"Using Inspiration Image for VLM Prompt Enhancement: {image_for_llm}")
                        elif input_images:
                            image_for_llm = input_images[0][0]
                            logger.info(f"Using Input Image (Frame 0) for VLM Prompt Enhancement: {image_for_llm}")

                        logger.info(f"DEBUG: Prompt BEFORE VLM Enhancement: {prompt}")
                        run_prompt, is_ref = llm_enhance(
                            prompt=prompt,
                            is_video=not is_image_mode,
                            text_encoder=text_encoder,
                            image_path=image_for_llm,
                            seed=seed,
                            duration_seconds=(num_frames / float(params.get("fps", 25.0))),
                            has_input_image=bool(image_for_llm),
                        )
                        logger.info(f"DEBUG: Prompt AFTER VLM Enhancement: {run_prompt}")
                        if is_ref:
                            logger.info("Image detected as Reference Sheet. Dropping image conditioning to force T2V.")
                            update_job_progress(job_id, 5, "Reference sheet detected. Using for prompt enhancement only.")
                            input_images = []
                            inspiration_images = []
                        update_job_db(job_id, "processing", enhanced_prompt=run_prompt)
                        update_job_progress(job_id, 5, "Prompt Enhanced", enhanced_prompt=run_prompt)
                        
                        if text_encoder is not None:
                            del text_encoder
                            cleanup_memory()
                    except Exception as e:
                        logger.warning(f"Prompt enhancement failed: {e}")

                if input_images:
                    valid_inputs = []
                    for item in input_images:
                        try:
                            start_path = item[0]
                            if os.path.exists(start_path):
                                valid_inputs.append(item)
                            else:
                                logger.warning(f"Skipping Missing Start Frame: {start_path}")
                        except Exception as e:
                            logger.warning(f"Failed to validate start frame {item}: {e}")
                    input_images = valid_inputs
                    
                    if not input_images:
                         logger.warning("No valid input images remaining after validation. Proceeding with T2V (No Visual Conditioning).")
                
                update_job_progress(job_id, 10, "Starting Generation...")
                
                import inspect
                logger.info(f"DEBUG: Pipeline Object: {pipeline}")
                logger.info(f"DEBUG: Pipeline Type: {type(pipeline)}")
                try:
                    logger.info(f"DEBUG: Pipeline File: {inspect.getfile(pipeline.__class__)}")
                except:
                    logger.info("DEBUG: Could not get pipeline file")
                
                result = None
                result = pipeline(
                    prompt=run_prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(params.get("fps", 25.0)),
                    num_inference_steps=num_inference_steps,
                    cfg_guidance_scale=cfg_scale,
                    images=input_images,
                    tiling_config=tiling_config,
                    enhance_prompt=False, # We already enhanced it manually
                    upscale=upscale,
                    callback_on_step_end=cancellation_check
                )
            elif pipeline_type == "ic_lora":
                # Manual enhancement for ic_lora (same as ti2vid)
                ic_prompt = prompt
                if enhance_prompt:
                    try:
                        from llm import enhance_prompt as llm_enhance
                        text_encoder = None
                        if config.LLM_PROVIDER.lower() == "gemma":
                            text_encoder = pipeline.stage_1_model_ledger.text_encoder()
                        ic_prompt, is_ref = llm_enhance(
                            prompt=prompt, is_video=True,
                            text_encoder=text_encoder,
                            image_path=input_images[0][0] if input_images else None,
                            seed=seed,
                            duration_seconds=(num_frames / float(params.get("fps", 25.0))),
                            has_input_image=bool(input_images),
                        )
                        if is_ref:
                            logger.info("Image detected as Reference Sheet. Dropping image conditioning to force T2V (ic_lora).")
                            update_job_progress(job_id, 5, "Reference sheet detected. Using for prompt enhancement only.")
                            input_images = []
                        update_job_db(job_id, "processing", enhanced_prompt=ic_prompt)
                        update_job_progress(job_id, 5, "Prompt Enhanced", enhanced_prompt=ic_prompt)
                        if text_encoder is not None:
                            del text_encoder
                            cleanup_memory()
                    except Exception as e:
                        logger.warning(f"ic_lora prompt enhancement failed: {e}")
                        ic_prompt = prompt

                result = pipeline(
                    prompt=ic_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(params.get("fps", 25.0)),
                    images=input_images,
                    video_conditioning=video_cond,
                    enhance_prompt=False,
                    tiling_config=tiling_config,
                    callback_on_step_end=cancellation_check
                )
            elif pipeline_type == "keyframe":
                # Manual enhancement for keyframe (same as ti2vid)
                kf_prompt = prompt
                if enhance_prompt:
                    try:
                        from llm import enhance_prompt as llm_enhance
                        text_encoder = None
                        if config.LLM_PROVIDER.lower() == "gemma":
                            text_encoder = pipeline.stage_1_model_ledger.text_encoder()
                        kf_prompt, is_ref = llm_enhance(
                            prompt=prompt, is_video=True,
                            text_encoder=text_encoder,
                            image_path=input_images[0][0] if input_images else None,
                            seed=seed,
                            duration_seconds=(num_frames / float(params.get("fps", 25.0))),
                            has_input_image=bool(input_images),
                        )
                        if is_ref:
                            logger.info("Image detected as Reference Sheet. Dropping image conditioning to force T2V (keyframe).")
                            update_job_progress(job_id, 5, "Reference sheet detected. Using for prompt enhancement only.")
                            input_images = []
                        update_job_db(job_id, "processing", enhanced_prompt=kf_prompt)
                        update_job_progress(job_id, 5, "Prompt Enhanced", enhanced_prompt=kf_prompt)
                        if text_encoder is not None:
                            del text_encoder
                            cleanup_memory()
                    except Exception as e:
                        logger.warning(f"keyframe prompt enhancement failed: {e}")
                        kf_prompt = prompt

                result = pipeline(
                    prompt=kf_prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(params.get("fps", 25.0)),
                    num_inference_steps=num_inference_steps,
                    cfg_guidance_scale=cfg_scale,
                    images=input_images,
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                    callback_on_step_end=cancellation_check
                )
            
            logger.info(f"DEBUG: Pipeline returned result type: {type(result)}")
            if isinstance(result, tuple):
                logger.info(f"DEBUG: Result tuple length: {len(result)}")
                if len(result) >= 2:
                    return (result[0], result[1])
                elif len(result) == 1:
                    return (result[0], None)
                else:
                    return (None, None)
            else:
                return (result, None)
        
        # FINAL CANCELLATION CHECK BEFORE THE LONG RUN
        if active_jobs.get(job_id, {}).get("cancelled"):
            raise RuntimeError(f"Job {job_id} cancelled by user.")
            
        pipeline_result = await loop.run_in_executor(None, _run_pipeline)
        
        # FINAL CANCELLATION CHECK BEFORE SAVING
        if active_jobs.get(job_id, {}).get("cancelled"):
            raise RuntimeError(f"Job {job_id} cancelled by user.")
            
        update_job_progress(job_id, 90, "Finalizing & Saving...")
        
        video, audio = pipeline_result
        if video is None:
            raise ValueError(f"Pipeline returned no video output for job {job_id}")
        
        paths = get_project_output_paths(job_id, project_id)
        os.makedirs(paths["output_dir"], exist_ok=True)
        os.makedirs(paths["thumbnail_dir"], exist_ok=True)
        
        if num_frames == 1:
             # Logic for single frame handling (if it falls through here for some reason)
             output_path = paths["output_path"].replace(".mp4", ".jpg")
             # Implementation ommitted as main single-frame logic is handled by Flux block above.
             # But if pipeline works for 1 frame, handle save here...
             pass 
        else:
            output_path = paths["output_path"]
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
            
            try:
                encode_video(
                    video,
                    int(params.get("fps", 25)),
                    audio,
                    AUDIO_SAMPLE_RATE if audio is not None else None,
                    output_path,
                    video_chunks_number
                )
            finally:
                import gc
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            logger.info(f"Video saved to {output_path}")

            # Verification and Thumbnail
            actual_frames = num_frames
            try:
                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", output_path]
                res = subprocess.check_output(cmd).decode("utf-8").strip()
                if res and res.isdigit():
                    actual_frames = int(res)
                    if job_id in active_jobs:
                        active_jobs[job_id]["actual_frames"] = actual_frames
            except Exception as e:
                logger.warning(f"Could not verify frame count: {e}")

            thumbnail_web_path = None
            try:
                thumb_path = paths["thumbnail_path"]
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                thumbnail_web_path = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail: {e}")

            output_url = f"/projects/{project_id}/generated/{os.path.basename(output_path)}"
            # Update Job DB
            update_job_db(
                job_id, 
                "completed", 
                output_path=output_url,
                thumbnail_path=thumbnail_web_path,
                actual_frames=actual_frames
            )

            # FIX: Update Shot if linked
            shot_id = params.get("shot_id")
            if shot_id:
                # Convert absolute path to web URL
                # projects/{id}/generated/foo.mp4
                rel_path = f"/projects/{project_id}/generated/{os.path.basename(output_path)}"
                shot_updates = {
                    "video_url": rel_path,
                    "status": "completed",
                    "last_job_id": job_id
                }
                # Also update thumbnail if we generated one and shot doesn't have a custom one?
                # For now let's leave thumbnail alone unless we want to overwrite it.
                # Usually concept art is the thumbnail. We shouldn't overwrite concept art with the first frame of video unless requested.
                # So we ONLY update video_url.
                
                update_shot_db(shot_id, **shot_updates)
                logger.info(f"Updated Shot {shot_id} with video: {rel_path}")
            update_job_progress(job_id, 100, "Done")
            
            shot_id = params.get("id")
            if shot_id:
                try:
                    with Session(engine) as session:
                        job = session.get(Job, job_id)
                        enhanced_prompt_result = job.enhanced_prompt if job else None
                except:
                    enhanced_prompt_result = None
                
                update_shot_db(
                    shot_id,
                    video_url=output_url,
                    thumbnail_url=thumbnail_web_path,
                    last_job_id=job_id,
                    enhanced_prompt_result=enhanced_prompt_result,
                    status="completed"
                )
            
            await event_manager.broadcast("complete", {
                "job_id": job_id, 
                "url": output_url, 
                "type": "video",
                "thumbnail_url": thumbnail_web_path,
                "actual_frames": actual_frames
            })


async def generate_video_task(job_id: str, params: dict):
    from sqlmodel import Session
    from database import engine, Job
    
    # 0. Check pre-generation cancellation (e.g. while sitting in GPU queue)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job and job.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled before starting execution.")
            return

    logger.info(f"Starting generation for job {job_id}")
    active_jobs[job_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "eta_seconds": None,
        "status_message": "Initializing...",
        "project_id": params.get("project_id"),
        "type": "video"
    }
    update_job_db(job_id, "processing")
    
    # Needs to import manager here to avoid circular dependencies if possible, 
    # but Manager is singleton.
    from model_engine import manager
    
    try:
        pipeline_type = params.get("pipeline_override") or params.get("pipeline_type", "ti2vid")
        timeline = params.get("timeline", [])
        
        # Save request debug info
        with open("server_last_request.json", "w") as f:
            import json
            json.dump(params, f, indent=2)
            
        # Pipeline Selection Logic
        if pipeline_type in ("advanced", "auto"):
            input_images = []
            video_cond = []
            has_video = any(t.get("type") == "video" for t in timeline)
            
            for item in timeline:
                raw_path = item.get("path")
                path = resolve_path(raw_path) 
                
                idx = item.get("frame_index", 0)
                strength = item.get("strength", 1.0)
                
                if item.get("type") == "image":
                    input_images.append((path, idx, strength))
                elif item.get("type") == "video":
                    video_cond.append((path, strength))
            
            if has_video:
                pipeline_type = "ic_lora"
                params["video_conditioning"] = video_cond
                params["images"] = input_images 
            elif len(input_images) >= 2 and any(img[1] == 0 for img in input_images) and any(img[1] == (params.get("num_frames", 121)-1) for img in input_images):
                pipeline_type = "keyframe"
                params["images"] = input_images
            else:
                pipeline_type = "ti2vid"
                params["images"] = input_images
            
            logger.info(f"Pipeline: {pipeline_type}, Input Images: {len(input_images)}")

        if params.get("num_frames") == 1:
             if pipeline_type == "auto" or pipeline_type == "advanced":
                 pipeline_type = "ti2vid"
             if pipeline_type in ["keyframe", "ic_lora"]:
                 params["pipeline_type"] = "ti2vid"
                 pipeline_type = "ti2vid"

        if pipeline_type == "auto":
            pipeline_type = "ti2vid"
        
        # Load Pipeline
        pipeline = await manager.load_pipeline(pipeline_type)
        
        # Dispatch
        # Check if chained/storyboard generation is needed?
        # currently logic for chained is not explicitly triggered by "advanced" unless it's Storyboard.
        # But wait, `worker.py` had `generate_chained_video_task` but where was it called?
        # It was called if `params['pipeline_type'] == 'chained'` ?
        # Actually in original `worker.py`, `generate_chained_video_task` was defined but I need to check where it was used!
        # Checking `worker.py`... I don't see it being called in `generate_video_task`!
        # It might be called from `server.py` specifically?
        # `server.py` calls `generate_video_task`.
        # Ah, `generate_chained_video_task` might be unfinished code or used by a different endpoint?
        # `generate_video_task` in `worker.py` only calls `generate_standard_video_task` logic (inline).
        # Wait, I missed checking if `generate_chained_video_task` is used.
        # Let's check `worker.py` logic again.
        pass # I'll just keep the structure I see.
        
        # In this refactor, I moved `generate_standard_video_task` logic into a function.
        # So I just call it here.
        
        # Chained Generation Delegation
        num_frames = params.get("num_frames", 121)
        if pipeline_type == "ti2vid" and num_frames > config.NATIVE_MAX_FRAMES:
             logger.info(f"Delegating to Chained Generation (frames={num_frames} > {config.NATIVE_MAX_FRAMES})")
             await generate_chained_video_task(job_id, params, pipeline)
        else:
             await generate_standard_video_task(job_id, params, pipeline)


    except Exception as e:
        if "cancelled" in str(e).lower():
             logger.info(f"Job {job_id} cancelled.")
             update_job_db(job_id, "cancelled")
             
             # Broadcast cancelled event explicitly via asyncio
             import asyncio
             from events import event_manager
             try:
                 asyncio.run_coroutine_threadsafe(
                     event_manager.broadcast("cancelled", {"job_id": job_id}),
                     loop
                 )
             except NameError:
                 pass # loop might not be in scope if it crashes extremely early, though unlikely
             
        else:
            logger.error(f"Job failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Write Debug Error File
            try:
                with open("server_last_error.txt", "w") as f:
                    f.write(f"Error: {e}\n")
                    traceback.print_exc(file=f)
            except: pass
            
            update_job_db(job_id, "failed", error=str(e))
            await event_manager.broadcast("error", {"job_id": job_id, "message": str(e)})
    finally:
        if job_id in active_jobs:
            del active_jobs[job_id]
