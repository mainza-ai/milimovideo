import asyncio
import os
import math
import inspect
import logging
import subprocess
import torch
import config
from job_utils import active_jobs, broadcast_log, broadcast_progress
from file_utils import get_project_output_paths
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.helpers import generate_enhanced_prompt, cleanup_memory

logger = logging.getLogger(__name__)

# Storyboard Manager Import
try:
    from backend.storyboard.manager import StoryboardManager
except ImportError:
    try:
        from storyboard.manager import StoryboardManager
    except ImportError:
        logger.error("Could not import StoryboardManager")
        StoryboardManager = None

async def generate_chained_video_task(job_id: str, params: dict, pipeline):
    """
    Handles autoregressive generation using StoryboardManager.
    """
    logger.info(f"Starting Storyboard chained generation for job {job_id}")
    
    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    seed = params.get("seed", 42)
    width = params.get("width", 768)
    height = params.get("height", 512)

    # Ensure mod64 for LTX-2
    height = round(height / 64) * 64
    width = round(width / 64) * 64
    # Total desired frames. The UI sends this based on selected duration.
    # e.g. 10s * 25fps = 250 frames.
    desired_total_frames = params.get("num_frames", 121) 
    
    # Get project-scoped output paths
    project_id = params.get("project_id", None)
    paths = get_project_output_paths(job_id, project_id)
    output_dir = paths["output_dir"]
    workspace_dir = paths["workspace_dir"]
    
    # Initialize Storyboard Manager with project workspace
    storyboard_manager = StoryboardManager(job_id, prompt, params, workspace_dir)
    
    chunk_size = storyboard_manager.chunk_size # 121
    num_chunks = storyboard_manager.get_total_chunks()
    
    # Initialize ETA immediately (45s per chunk heuristic)
    if job_id in active_jobs:
        active_jobs[job_id]["eta_seconds"] = num_chunks * 45
        active_jobs[job_id]["status_message"] = f"Initializing generation ({num_chunks} chunks)..."
    
    logger.info(f"Chained generation: {desired_total_frames} frames split into {num_chunks} chunks.")
    
    files_to_stitch = []

    # Helper to broadcast async (can be called from main loop)
    async def update_ui(msg: str):
         await broadcast_log(job_id, msg)

    loop = asyncio.get_running_loop()
    
    last_chunk_output = None
    
    try:
        last_chunk_latent = None
        
        for chunk_idx in range(num_chunks):
            # Check for cancellation before starting chunk
            if active_jobs.get(job_id, {}).get("cancelled", False):
                raise RuntimeError(f"Job {job_id} cancelled by user.")

            chunk_job_id = f"{job_id}_part_{chunk_idx}"
            chunk_output_path = os.path.join(output_dir, f"{chunk_job_id}.mp4")
            files_to_stitch.append(chunk_output_path)
            
            # Prepare cancellation (and progress updates)
            # Defined INSIDE loop to access chunk_idx for global progress
            def cancellation_check(step_idx, total_steps, *args):
                 if job_id in active_jobs:
                     chunk_progress = step_idx / total_steps
                     
                     # 1. Global Progress
                     global_progress = int(((chunk_idx + chunk_progress) / num_chunks) * 100)
                     active_jobs[job_id]["progress"] = global_progress
                     
                     # 2. Dynamic ETA
                     # Remaining full chunks + remaining time in current chunk
                     remaining_chunks = num_chunks - chunk_idx - 1
                     current_chunk_remaining = 45 * (1 - chunk_progress)
                     eta = int((remaining_chunks * 45) + current_chunk_remaining)
                     active_jobs[job_id]["eta_seconds"] = eta
                     
                     # 3. Status Message
                     active_jobs[job_id]["status_message"] = f"Generating Chunk {chunk_idx+1}/{num_chunks} ({int(chunk_progress*100)}%)"

                 if active_jobs.get(job_id, {}).get("cancelled", False):
                     raise RuntimeError(f"Job {job_id} cancelled by user.")

            await update_ui(f"Generating chunk {chunk_idx+1}/{num_chunks}...")
            # Calculate roughly global progress? 
            # Chunk progress is local. Global = (chunk_idx / num_chunks) * 100.
            await broadcast_progress(job_id, int((chunk_idx / num_chunks) * 100))
            
            # 1. Prepare configuration via Manager
            # We pass the text encoder here so the manager can use it for prompt enhancement
            text_encoder = None
            # Force auto_continue if enhance_prompt is active (Smart Prompt Evolution)
            should_auto_continue = params.get("auto_continue", False) or params.get("enhance_prompt", False)

            if chunk_idx > 0 and should_auto_continue and hasattr(pipeline, "stage_1_model_ledger"):
                 logger.info("Retrieving text encoder for narrative prompt generation...")
                 try:
                     text_encoder = pipeline.stage_1_model_ledger.text_encoder()
                 except Exception as e:
                     logger.warning(f"Could not retrieve text encoder: {e}")

            chunk_config = await storyboard_manager.prepare_next_chunk(chunk_idx, last_chunk_output, text_encoder)
            
            # Cleanup encoder reference to free memory if needed (though manager just used it)
            if text_encoder:
                del text_encoder
                cleanup_memory()
            
            # Extract params
            chunk_prompt = chunk_config.get("prompt", prompt)
            chunk_images = chunk_config.get("images", []) # list of (path, idx, strength)
            overlap_count = 0

            # PREPARE LATENT CONDITIONING for Handoff (QUANTUM ALIGNMENT FIX)
            conditioning_latent_tensor = None
            frames_to_trim = 0 # Track exact number of frames to trim later
            
            if last_chunk_latent is not None:
                # LTX-2 Temporal Downsample Factor is 8.
                # We typically interpret 'overlap' as pixel frames (e.g., 24).
                # We must convert this to 'latent frames', but robustly.
                # Formula: latent_slices = ceil((overlap_pixels - 1) / 8) + 1
                # Example: 24 pixels -> ceil(23/8) + 1 = 3 + 1 = 4 latent slices.
                
                requested_pixel_overlap = len(chunk_images)
                overlap_count = requested_pixel_overlap # Keep for reference
                
                if requested_pixel_overlap > 0:
                    # 1. Calculate Latent Slices needed to cover these pixels
                    latent_slice_count = math.ceil((requested_pixel_overlap - 1) / 8) + 1
                    
                    # 2. Back-calculate the EFFECTIVE pixel overlap this represents
                    # (latent_slice_count - 1) * 8 + 1
                    effective_overlap_pixels = (latent_slice_count - 1) * 8 + 1
                    frames_to_trim = effective_overlap_pixels
                    
                    # 3. Slice the Latent Tensor
                    # last_chunk_latent shape: [1, C, F, H, W]
                    total_prev_latents = last_chunk_latent.shape[2]
                    
                    # Safety clamp
                    latent_slice_count = min(latent_slice_count, total_prev_latents)
                    
                    # Extract last N latent frames
                    # conditioning_latent_tensor = last_chunk_latent[:, :, -latent_slice_count:, :, :].clone()
                    start_slice = total_prev_latents - latent_slice_count
                    conditioning_latent_tensor = last_chunk_latent[:, :, start_slice:, :, :].clone()
                    
                    logger.info(f"Latent Handoff: Requested {requested_pixel_overlap} px -> Aligning to {latent_slice_count} Latents ({effective_overlap_pixels} px effective).")
                    
                    # CONFLICT RESOLUTION (Static Anchor Fix):
                    # If we have successfully prepared High-Fidelity Latent Handoff (Motion),
                    # we must DISCARD the Static Pixel Conditioning (Images).
                    # Otherwise, the static images "anchor" the generation, freezing the first second.
                    if conditioning_latent_tensor is not None:
                        logger.info("Latent Handoff Active: Disabling Static Pixel Conditioning to prevent 'Frozen Anchor' effect.")
                        chunk_images = [] 


            # Critical Fix for Extend + Smart Continue:
            # If this is the first chunk, and we have global input images (from Extend timeline),
            # we must use them! Storyboard doesn't know about them yet (or returned empty).
            if chunk_idx == 0 and not chunk_images and params.get("images"):
                 chunk_images = params.get("images")
                 logger.info(f"Using initial timeline images for Chunk 0: {len(chunk_images)}")

            # Manually handle Prompt Enhancement to inject "Director" logic
            # Update: delegated to StoryboardManager.prepare_next_chunk
            # We only need to ensure we use the prompt it returns.
            do_pipeline_enhance = False 

            # Only enhance manually for Chunk 0 if it wasn't handled (e.g. Extend mode needs Director prompt)
            if chunk_idx == 0 and params.get("enhance_prompt", True):
                if not text_encoder and hasattr(pipeline, "stage_1_model_ledger"):
                     try:
                         text_encoder = pipeline.stage_1_model_ledger.text_encoder()
                     except: pass
                
                if text_encoder:
                     # Detect extend mode
                     is_extend = (chunk_idx == 0 and chunk_images)
                     sys_prompt = None
                     effective_prompt = chunk_prompt
                     
                     if is_extend:
                          # Use the same STRICT Director Prompt as StoryboardManager to prevent drift!
                          sys_prompt = (
                             "You are a visionary Film Director and Cinematographer. Your goal is to continue the narrative flow of a video scene.\n"
                             "You will be given the Global Story Goal (which you must adhere to) and the immediate context (last frame description).\n"
                             "TASK: Describe the next 4 seconds of video action. The transition must be seamless but FOCUSED on the Global Goal.\n"
                             "GUIDELINES:\n"
                             "- Analyze the visual context of the input image (characters, clothing, lighting, background).\n"
                             "- Describe the ACTION that happens next. Do not just describe the static image.\n"
                             "- CRITICAL: Do not drift from the Global Story Goal. If the context has drifted, steer it back.\n"
                             "- Maintain character identity and visual consistency.\n"
                             "- CINEMATOGRAPHY: Specify camera movement (e.g., 'slow dolly in', 'pan right', 'handheld', 'static').\n"
                             "- LIGHTING: Describe the lighting atmosphere (e.g., 'cinematic lighting', 'soft morning light', 'neon rim light').\n"
                             "- AUDIO: Include a description of the soundscape (ambient sounds, dialogue if applicable).\n"
                             "- Output ONLY the prompt for the next shot. Single paragraph, chronological flow."
                          )
                          effective_prompt = (
                             f"Global Story Goal (PRIMARY): {chunk_prompt}. "
                             f"Task: Write a prompt for the NEXT 4 seconds of video extending the provided frame. "
                             f"The action MUST advance the Global Story Goal."
                          )

                     try:
                         logger.info(f"Manually enhancing Chunk 0 prompt (Extend={is_extend})...")
                         enhanced = generate_enhanced_prompt(
                             text_encoder, 
                             effective_prompt, 
                             image_path=chunk_images[0][0] if chunk_images else None,
                             seed=seed,
                             is_image=False, # Video generation
                             system_prompt=sys_prompt
                         )
                         if enhanced:
                             chunk_prompt = enhanced
                             active_jobs[job_id]["current_prompt"] = chunk_prompt
                             
                             # CRITICAL FIX: Update the Global Story Goal!
                             logger.info(f"Updating Global Story Goal to match enhanced Chunk 0 prompt...")
                             storyboard_manager.state.global_prompt = enhanced
                     except Exception as e:
                         logger.warning(f"Manual enhancement failed: {e}")
                         do_pipeline_enhance = True # Fallback to pipeline
                
            
            # Expose current prompt to UI
            if job_id in active_jobs:
                active_jobs[job_id]["current_prompt"] = chunk_prompt
                # Determine progress
                remaining_chunks = num_chunks - chunk_idx
                # Heuristic: 45 seconds per chunk
                eta_seconds = remaining_chunks * 45 
                active_jobs[job_id]["eta_seconds"] = eta_seconds
                
                # Also status message
                active_jobs[job_id]["status_message"] = f"Generating Chunk {chunk_idx+1}/{num_chunks}"
            
            logger.info(f"Starting chunk {chunk_idx}: {chunk_prompt[:50]}...")
            
            # Check cancellation again before expensive call
            if job_id not in active_jobs or active_jobs[job_id].get("cancelled", False):
                break
            
            # Update status
            active_jobs[job_id]["status_message"] = f"Generating Chunk {chunk_idx+1}/{num_chunks}"

            # 2. Define Pipeline execution wrapper
            def _run_chunk_pipeline(images_arg, current_prompt, pipeline_enhance, latents_in):
                # Force Negative Prompt for chained chunks to kill static inertia
                # We start with the user's negative prompt (if any) and append our bans.
                effective_negative_prompt = negative_prompt or ""
                if chunk_idx > 0:
                     effective_negative_prompt += ", static, freeze, loop, pause, still image, motionless, blurred, morphing"
                
                return pipeline(
                    prompt=current_prompt,
                    negative_prompt=effective_negative_prompt,
                    seed=seed + chunk_idx, # vary seed slightly
                    height=height,
                    width=width,
                    num_frames=chunk_size,
                    frame_rate=float(params.get("fps", 25.0)),
                    num_inference_steps=params.get("num_inference_steps", 40),
                    cfg_guidance_scale=params.get("cfg_scale", 3.0),
                    images=images_arg,
                    previous_latent_tensor=latents_in, # New Arg
                    tiling_config=TilingConfig.default(),
                    enhance_prompt=pipeline_enhance,
                    callback_on_step_end=cancellation_check
                )
                
            # 3. Run Pipeline
            # Returns (video, audio, latent)
            try:
                video, audio, new_full_latent = await loop.run_in_executor(
                    None, _run_chunk_pipeline, chunk_images, chunk_prompt, do_pipeline_enhance, conditioning_latent_tensor
                )
                
                # Update cache for next iteration
                last_chunk_latent = new_full_latent
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                raise e
            
            # 4. Save Output
            
            # LATENT HANDOFF: TRIM OVERLAP
            # If we used latent conditioning from the previous chunk, the model generated
            # those frames at the start. We must remove them to avoid "skip back" in playback.
            # QUANTUM FIX: Use 'frames_to_trim' which is aligned to latent boundaries.
            if chunk_idx > 0 and conditioning_latent_tensor is not None and frames_to_trim > 0:
                logger.info(f"Trimming {frames_to_trim} overlap frames from start of Chunk {chunk_idx}...")
                
                # Helper for generator trimming
                def trim_video_iterator(iterator, n_trim):
                    frames_trimmed = 0
                    for chunk in iterator:
                        # chunk shape: [F, H, W, C]
                        n_frames = chunk.shape[0]
                        
                        if frames_trimmed < n_trim:
                            remaining_to_trim = n_trim - frames_trimmed
                            if n_frames <= remaining_to_trim:
                                # Drop entire chunk
                                frames_trimmed += n_frames
                                continue
                            else:
                                # Partial drop
                                yield chunk[remaining_to_trim:]
                                frames_trimmed = n_trim # Done
                        else:
                            yield chunk

                # Trim Video
                if inspect.isgenerator(video):
                    video = trim_video_iterator(video, frames_to_trim)
                elif isinstance(video, torch.Tensor):
                    # Expecting [F, H, W, C] from decode_video output convention
                    video = video[frames_to_trim:]
                
                # Trim Audio: [B, C, Samples] or [C, Samples]
                if audio is not None:
                     fps = float(params.get("fps", 25.0))
                     trim_samples = int(frames_to_trim * (AUDIO_SAMPLE_RATE / fps))
                     if audio.ndim == 3:
                        audio = audio[:, :, trim_samples:]
                     elif audio.ndim == 2:
                        audio = audio[:, trim_samples:]
            
            video_chunks_number = get_video_chunks_number(chunk_size, TilingConfig.default())
            encode_video(
                video=video,
                fps=float(params.get("fps", 25.0)),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=chunk_output_path,
                video_chunks_number=video_chunks_number,
            )
            
            # 5. Commit to Manager (updating state)
            await storyboard_manager.commit_chunk(chunk_idx, chunk_output_path, chunk_prompt)
            last_chunk_output = chunk_output_path
            
            # Cleanup memory
            del video
            del audio
            cleanup_memory()

            # TRIM OVERLAP (Fix for "Slow Motion" stutter at seams)
            # If this is not the first chunk, we must trim the overlapping frames 
            if chunk_idx > 0:
                 # Use global overlap setting from manager
                overlap = storyboard_manager.overlap_frames
                if overlap > 0:
                    logger.info(f"Trimming {overlap} overlap frames from chunk {chunk_idx} start...")
                    temp_raw_path = chunk_output_path.replace(".mp4", "_raw.mp4")
                    os.rename(chunk_output_path, temp_raw_path)
                    
                    try:
                        # Use select filter to drop first N frames. 
                        # -vsync vfr ensures frames are dropped correctly without duping, 
                        # but we want to ensure the output is compatible with the target fps.
                        # ADDED: afade to smooth the audio join (fadeIn 0.1s)
                        fps = params.get("fps", 25.0)
                        
                        subprocess.run([
                            "ffmpeg", "-i", temp_raw_path,
                            "-vf", f"select=gte(n\\,{overlap}),setpts=PTS-STARTPTS",
                            "-af", f"ashowinfo,arealtime,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.1", 
                            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            "-r", str(fps),
                            "-y", chunk_output_path
                        ], cwd=output_dir, check=True, stderr=subprocess.DEVNULL)
                        logger.info(f"Trimmed overlap from chunk {chunk_idx} (with Audio Fade).")
                    except Exception as e:
                        logger.error(f"Failed to trim chunk overlap: {e}. Restoring raw.")
                        if os.path.exists(temp_raw_path):
                            os.rename(temp_raw_path, chunk_output_path)
            
        # Final stitching complete
        logger.info(f"All chunks generated successfully. Stitching {len(files_to_stitch)} parts...")
        
        # Ensure output directories exist
        os.makedirs(paths["output_dir"], exist_ok=True)
        os.makedirs(paths["thumbnail_dir"], exist_ok=True)
        
        # Use project-scoped final output path
        final_output_path = paths["output_path"]
        
        # Stitch videos using ffmpeg
        concat_list = os.path.join(workspace_dir, f"{job_id}_concat_list.txt")
        # We re-encode to avoid codec mismatches between original chunks and trimmed chunks.
        input_args = []
        filter_parts = []
        
        for i, mp4 in enumerate(files_to_stitch):
            input_args.extend(["-i", mp4])
            filter_parts.append(f"[{i}:v][{i}:a]")
            
        # Construct filter string: "[0:v][0:a][1:v][1:a]concat=n=N:v=1:a=1[outv][outa]"
        filter_complex = f"{''.join(filter_parts)}concat=n={len(files_to_stitch)}:v=1:a=1[outv][outa]"
        
        cmd = ["ffmpeg"] + input_args + [
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", # Good quality re-encode
            "-c:a", "aac", "-b:a", "192k",
            "-y", final_output_path
        ]
        
        try:
            subprocess.run(cmd, cwd=output_dir, check=True)
            logger.info(f"Stitched video saved to {final_output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Stitching failed with audio: {e}. Retrying video only...")
            # Fallback: Video only if audio mapping fails (e.g. some chunks have no audio)
            filter_parts_v = [f"[{i}:v]" for i in range(len(files_to_stitch))]
            filter_complex_v = f"{''.join(filter_parts_v)}concat=n={len(files_to_stitch)}:v=1[outv]"
            cmd_v = ["ffmpeg"] + input_args + [
                "-filter_complex", filter_complex_v,
                "-map", "[outv]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-y", final_output_path
            ]
            subprocess.run(cmd_v, cwd=output_dir, check=True)
            logger.info(f"Stitched video (video-only) saved to {final_output_path}")
        
        # Detect actual frame count (DURATION PROBE) to sync UI for Chained Generation
        actual_frames = desired_total_frames
        try:
            # Probe the file DURATION (more accurate for player sync than packet count)
            # -show_entries format=duration
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", final_output_path]
            res = subprocess.check_output(cmd).decode("utf-8").strip()
            
            if res:
                duration_sec = float(res)
                fps = float(params.get("fps", 25.0))
                actual_frames_calc = int(round(duration_sec * fps))
                
                logger.info(f"Verified actual duration: {duration_sec}s -> {actual_frames_calc} frames (UI Sync)")
                actual_frames = actual_frames_calc
                
                if job_id in active_jobs:
                    active_jobs[job_id]["actual_frames"] = actual_frames
        except Exception as e:
            logger.warning(f"Could not verify chained duration: {e}")

        # GENERATE THUMBNAIL (For Timeline UI)
        # Generate thumbnail from final video
        thumb_path = paths["thumbnail_path"]
        thumbnail_web_path = None
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", final_output_path, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Generated thumbnail: {thumb_path}")
            thumbnail_web_path = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}" if project_id else thumb_path
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")

        
        cleanup_memory()
        
        # Convert to project-relative URLs for database
        if project_id:
            relative_output = f"/projects/{project_id}/generated/{os.path.basename(final_output_path)}"
            relative_thumb = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}" if thumb_path else None
        else:
            # Legacy paths (backward compatibility)
            relative_output = final_output_path
            relative_thumb = thumb_path
        
        # Update job record
        update_job_db(
            job_id, 
            "completed", 
            output_path=relative_output,
            enhanced_prompt=chunk_prompt, # Using last chunk prompt as enhanced? Or we should aggregate?
            actual_frames=actual_frames,
            thumbnail_path=relative_thumb
        )
        
        await event_manager.broadcast("complete", {
            "job_id": job_id, 
            "url": relative_output,  # Use project-scoped URL
            "type": "video",
            "actual_frames": actual_frames,
            "thumbnail_url": relative_thumb  # Use project-scoped URL
        })
        
    finally:
        # Cleanup split files and artifacts
        if not active_jobs.get(job_id, {}).get("cancelled", False):
            storyboard_manager.cleanup()
