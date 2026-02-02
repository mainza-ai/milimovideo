import asyncio
import os
import subprocess
import torch
import sys
import uuid
import logging
import gc
import json
from typing import Optional, Any, Dict
import math
import inspect 
import config
from config import PROJECTS_DIR

# Setup paths
config.setup_paths()

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_pipelines.utils.helpers import generate_enhanced_prompt, cleanup_memory
from PIL import Image
import numpy as np

try:
    from backend.storyboard.manager import StoryboardManager
except ImportError:
    from storyboard.manager import StoryboardManager

from events import event_manager
from database import engine, Job
from sqlmodel import Session
from datetime import datetime, timezone

# Configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_job_db(job_id: str, status: str, output_path: str = None, error: str = None, enhanced_prompt: str = None, status_message: str = None, actual_frames: int = None, thumbnail_path: str = None):
    """Helper to update Job record in SQLite."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = status
                job.params_json = job.params_json
                if output_path: 
                    job.output_path = output_path
                if error: 
                    job.error_message = str(error)
                if enhanced_prompt:
                    job.enhanced_prompt = enhanced_prompt
                if status_message:
                    job.status_message = status_message
                if actual_frames is not None:
                    job.actual_frames = actual_frames
                if thumbnail_path:
                    job.thumbnail_path = thumbnail_path
                if status in ["completed", "failed", "cancelled"]:
                    job.completed_at = datetime.now(timezone.utc)
                session.add(job)
                session.commit()
    except Exception as e:
        logger.error(f"Failed to update job DB for {job_id}: {e}")

def resolve_element_image_path(image_path: str) -> str | None:
    """Resolve element image_path (web URL format) to absolute filesystem path.
    
    Element visuals are stored as web URLs (e.g., /projects/{id}/assets/elements/...)
    but we need absolute paths for image loading.
    """
    if not image_path:
        return None
    
    # Already an absolute path that exists
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # Web URL format: /projects/{id}/assets/...
    if image_path.startswith("/projects"):
        relative = image_path.lstrip("/projects/")  # -> {id}/assets/elements/...
        full_path = os.path.join(PROJECTS_DIR, relative)
        if os.path.exists(full_path):
            return full_path
        else:
            logger.warning(f"Element image not found at resolved path: {full_path}")
    
    return None

async def broadcast_log(job_id: str, message: str):
    logger.info(f"[{job_id}] {message}")
    await event_manager.broadcast("log", {"job_id": job_id, "message": message})
    
async def broadcast_progress(job_id: str, progress: int, status: str = "processing"):
    await event_manager.broadcast("progress", {"job_id": job_id, "progress": progress, "status": status})

# ... (ModelManager class) ...

# We need to inject update_job_db calls into the sub-functions.
# Instead of replacing the WHOLE file, let's use replace_file_content carefully on specific blocks.


class ModelManager:
    def __init__(self):
        self._pipeline = None
        self._pipeline_type = None
        self._lock = asyncio.Lock()
        
    def get_base_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def get_model_paths(self):
        # Use centralized config
        ltx2_root = config.LTX_DIR
        models_dir = os.path.join(ltx2_root, "models")
        
        # Auto-select best available checkpoint
        # Priority 1: Full Precision / BF16 (Better for MPS/CPU stability if available)
        ckpt_full = os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled.safetensors")
        # Priority 2: FP8 (Smaller, but requires cast on MPS)
        ckpt_fp8 = os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-fp8.safetensors")
        
        selected_ckpt = ckpt_full
        if os.path.exists(ckpt_full):
            logger.info(f"Selected Main Checkpoint: {ckpt_full}")
        elif os.path.exists(ckpt_fp8):
            selected_ckpt = ckpt_fp8
            logger.info(f"Selected FP8 Checkpoint: {ckpt_fp8}")
        else:
            logger.warning("No checkpoint found! Defaulting to full path logic.")

        return {
            "checkpoint_path": selected_ckpt,
            "distilled_lora_path": os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-lora-384.safetensors"), 
            "spatial_upsampler_path": os.path.join(models_dir, "upscalers", "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
            "temporal_upsampler_path": os.path.join(models_dir, "upscalers", "ltx-2-temporal-upscaler-x2-1.0.safetensors"),
            "gemma_root": os.path.join(models_dir, "text_encoders", "gemma3"),
            # Placeholder for IP-Adapters
            "flux_ip_adapter": config.FLUX_IP_ADAPTER_PATH,
        }

    async def load_pipeline(self, pipeline_type: str, loras: list = None):
        if loras is None:
            loras = []

        async with self._lock:
            # Check if we already have this pipeline loaded
            # Note: For strict correctness, we should also check if loras changed, 
            # but for now we assume loras are mostly static or we reload if needed.
            # Simpler: just check type for now.
            if self._pipeline is not None and self._pipeline_type == pipeline_type:
                logger.info(f"Pipeline {pipeline_type} already loaded.")
                return self._pipeline

            # Unload existing
            if self._pipeline is not None:
                logger.info("Unloading previous pipeline...")
                del self._pipeline
                self._pipeline = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            logger.info(f"Loading LTX-2 Pipeline: {pipeline_type}...")
            
            paths = self.get_model_paths()
            
            # Check device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            logger.info(f"Using device: {device}")
            
            # Common args
            distilled_lora_objs = []
            if os.path.exists(paths["distilled_lora_path"]):
                distilled_lora_objs.append(
                    LoraPathStrengthAndSDOps(paths["distilled_lora_path"], 1.0, None)
                )

            is_mps = (device == "mps")
            fp8 = False if is_mps else True 
            
            if pipeline_type == "ti2vid":
                self._pipeline = TI2VidTwoStagesPipeline(
                    checkpoint_path=paths["checkpoint_path"],
                    distilled_lora=distilled_lora_objs,
                    spatial_upsampler_path=paths["spatial_upsampler_path"],
                    temporal_upsampler_path=paths["temporal_upsampler_path"],
                    gemma_root=paths["gemma_root"],
                    loras=loras,
                    device=device,
                    fp8transformer=fp8
                )
            elif pipeline_type == "ic_lora":
                # ICLoraPipeline takes 'loras' as the specific IC-LoRA
                self._pipeline = ICLoraPipeline(
                    checkpoint_path=paths["checkpoint_path"],
                    spatial_upsampler_path=paths["spatial_upsampler_path"],
                    gemma_root=paths["gemma_root"],
                    loras=loras, # These will be the IC-LoRAs
                    device=device,
                    fp8transformer=fp8
                )
            elif pipeline_type == "keyframe":
                self._pipeline = KeyframeInterpolationPipeline(
                    checkpoint_path=paths["checkpoint_path"],
                    distilled_lora=distilled_lora_objs,
                    spatial_upsampler_path=paths["spatial_upsampler_path"],
                    gemma_root=paths["gemma_root"],
                    loras=loras,
                    device=device,
                    fp8transformer=fp8
                )
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            self._pipeline_type = pipeline_type
            
            # MPS VAE Fix: Force Float32 to prevent black output
            if device == "mps" and hasattr(self._pipeline, "vae"):
                logger.info("MPS detected: Converting VAE to float32 to prevent black output...")
                self._pipeline.vae = self._pipeline.vae.to(dtype=torch.float32)

            logger.info(f"LTX-2 Pipeline {pipeline_type} loaded successfully.")
            return self._pipeline

# Singleton instance
manager = ModelManager()

def get_base_dir():
    """Get backend base directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_project_output_paths(job_id: str, project_id: str = None):
    """
    Get output paths for a generation job.
    If project_id is provided, uses project workspace.
    Otherwise falls back to legacy generated/ folder.
    """
    base_dir = get_base_dir()
    
    if project_id:
        # Project-scoped paths
        projects_dir = os.path.join(base_dir, "projects", project_id)
        return {
            "output_dir": os.path.join(projects_dir, "generated"),
            "thumbnail_dir": os.path.join(projects_dir, "thumbnails"),
            "workspace_dir": os.path.join(projects_dir, "workspace"),
            "output_path": os.path.join(projects_dir, "generated", f"{job_id}.mp4"),
            "thumbnail_path": os.path.join(projects_dir, "thumbnails", f"{job_id}_thumb.jpg"),
            "project_id": project_id
        }
    else:
        # Legacy global paths (for backward compatibility)
        generated_dir = os.path.join(base_dir, "generated")
        return {
            "output_dir": generated_dir,
            "thumbnail_dir": generated_dir,
            "workspace_dir": generated_dir,
            "output_path": os.path.join(generated_dir, f"{job_id}.mp4"),
            "thumbnail_path": os.path.join(generated_dir, f"{job_id}_thumb.jpg"),
            "project_id": None
        }

# --- Pipeline Cache ---
# Cancellation tracking
# job_id -> {'cancelled': bool}
active_jobs: Dict[str, Dict[str, Any]] = {}

# Helper to get the next multiple of 64 or 32 as needed, though defaults should be safe.
# We will use simple concatenation logic.

import subprocess

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
    output_dir = paths["output_dir"] # Fix NameError
    
    # Initialize Storyboard Manager with project workspace
    storyboard_manager = StoryboardManager(job_id, prompt, params, paths["workspace_dir"])
    
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
        concat_list = os.path.join(paths["workspace_dir"], f"{job_id}_concat_list.txt")
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
                "ffmpeg", "-y", "-i", final_output, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Generated thumbnail: {thumb_path}")
            thumbnail_web_path = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}" if project_id else thumb_path
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")

        
        cleanup_memory()
        
        # Convert to project-relative URLs for database
        if project_id:
            relative_output = f"/projects/{project_id}/generated/{os.path.basename(final_output)}"
            relative_thumb = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}" if thumb_path else None
        else:
            # Legacy paths (backward compatibility)
            relative_output = final_output
            relative_thumb = thumb_path
        
        # Update job record
        update_job_db(
            job_id, 
            "completed", 
            output_path=relative_output,
            enhanced_prompt=final_prompt,
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


async def generate_standard_video_task(job_id: str, params: dict, pipeline):
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        seed = params.get("seed", 42)
        width = params.get("width", 768)
        height = params.get("height", 512)
        
        # Ensure mod64 for LTX-2
        height = round(height / 64) * 64
        width = round(width / 64) * 64
        num_frames = params.get("num_frames", 121) 
        num_inference_steps = params.get("num_inference_steps", 40)
        cfg_scale = params.get("cfg_scale", 3.0)
        enhance_prompt = params.get("enhance_prompt", True)
        input_images = params.get("images", [])
        
        # RESOLVE WEB PATHS for Timeline Input Images
        # Structure: [(path, idx, strength)]
        # Converts /projects/... -> /Users/.../backend/projects/...
        if input_images:
            resolved_inputs = []
            base_dir = get_base_dir()
            for item in input_images:
                # Handle potential unpacking issues if item structure varies, but assumed tuple/list
                try:
                    path, idx, strength = item
                    if isinstance(path, str) and path.startswith("/projects"):
                        abs_path = os.path.join(base_dir, path.lstrip("/"))
                        if os.path.exists(abs_path):
                            resolved_inputs.append((abs_path, idx, strength))
                        else:
                            logger.warning(f"Timeline Image Not Found & Skipped: {abs_path}")
                            # SKIP adding it to resolved_inputs so pipeline ignores it
                    else:
                        if os.path.exists(path) if isinstance(path, str) else True:
                            resolved_inputs.append(item)
                except Exception as e:
                    logger.warning(f"Failed to resolve input image path: {item} - {e}")
                    resolved_inputs.append(item)
            input_images = resolved_inputs

        element_images = params.get("element_images", [])

        # RESOLVE WEB PATHS for Element Images
        # Converts /projects/... -> /Users/.../backend/projects/...
        if element_images:
            resolved_elements = []
            base_dir = get_base_dir()
            for img_path in element_images:
                if isinstance(img_path, str) and img_path.startswith("/projects"):
                    abs_path = os.path.join(base_dir, img_path.lstrip("/"))
                    if os.path.exists(abs_path):
                        resolved_elements.append(abs_path)
                    else:
                        logger.warning(f"Element Image Not Found & Skipped: {abs_path}")
                        # SKIP
                else:
                    if os.path.exists(img_path) if isinstance(img_path, str) else True:
                        resolved_elements.append(img_path)
            element_images = resolved_elements
        
        # Visual Conditioning Logic (Phase 1.5)
        # If no explicit timeline images, use Element Visuals as Start Frame
        is_inferred_start_frame = False
        if not input_images and element_images:
            logger.info(f"Applying Visual Conditioning from Elements: {len(element_images)} images.")
            # Use first element image as Start Frame (idx=0, str=1.0)
            input_images = [(element_images[0], 0, 1.0)]
            is_inferred_start_frame = True

        video_cond = params.get("video_conditioning", [])
        pipeline_type = params.get("pipeline_type", "ti2vid")
        project_id = params.get("project_id", None)  # Extract project_id for workspace paths
        
        # Initialize ETA
        if job_id in active_jobs:
            # Heuristic: 45s for video, 5s for image
            est_time = 45 if num_frames > 1 else 5
            active_jobs[job_id]["eta_seconds"] = est_time
            active_jobs[job_id]["status_message"] = "Initializing generation..."
        
        
        loop = asyncio.get_running_loop()
        
        if num_frames == 1:
            logger.info(f"Detected single-frame generation. Delegating to Flux 2 (T2I mode)...")
            from models.flux_wrapper import flux_inpainter
            
            # Use executor to run blocking Flux generation
            def _run_flux():
                 def flux_callback(step, total):
                     if job_id in active_jobs:
                         active_jobs[job_id]["progress"] = int((step / total) * 100)
                         active_jobs[job_id]["status_message"] = f"Generating Image ({step}/{total})"

                 img = flux_inpainter.generate_image(
                     prompt=prompt,
                     width=width,
                     height=height,
                     guidance=cfg_scale, # Flux guidance
                     ip_adapter_images=element_images, # Pass visual conditioning 
                     callback=flux_callback
                 )
                 # Save
                 out_filename = f"{job_id}.jpg"
                 
                 # Project scoped path?
                 if project_id:
                     projects_dir = os.path.join(get_base_dir(), "projects", project_id)
                     out_path = os.path.join(projects_dir, "generated", out_filename)
                     # Ensure dirs
                     os.makedirs(os.path.dirname(out_path), exist_ok=True)
                     
                     # Also thumb
                     thumb_dir = os.path.join(projects_dir, "thumbnails")
                     os.makedirs(thumb_dir, exist_ok=True)
                     thumb_path = os.path.join(thumb_dir, f"{job_id}_thumb.jpg")
                     
                     # Web paths
                     web_url = f"/projects/{project_id}/generated/{out_filename}"
                     web_thumb = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
                 else:
                     # Legacy
                     out_path = os.path.join(os.path.dirname(__file__), "generated", out_filename)
                     thumb_path = out_path.replace(".jpg", "_thumb.jpg")
                     web_url = f"/generated/{out_filename}"
                     web_thumb = web_url.replace(".jpg", "_thumb.jpg")
                 
                 img.save(out_path, quality=95)
                 img.resize((round(width/4), round(height/4))).save(thumb_path)
                 
                 return web_url, web_thumb

            try:
                web_url, web_thumb = await loop.run_in_executor(None, _run_flux)
                
                logger.info(f"Flux Image Generated: {web_url}")
                
                update_job_db(
                    job_id, 
                    "completed", 
                    output_path=web_url, 
                    thumbnail_path=web_thumb,
                    actual_frames=1,
                    status_message="Flux Image Ready"
                )
                
                await event_manager.broadcast("complete", {
                    "job_id": job_id, 
                    "url": web_url,
                    "type": "image",
                    "thumbnail_url": web_thumb
                })
                return
                
            except Exception as e:
                logger.error(f"Flux Generation failed: {e}")
                update_job_db(job_id, "failed", error=str(e))
                await event_manager.broadcast("error", {"job_id": job_id, "message": str(e)})
                return

        tiling_config = TilingConfig.default()
        
        # Progress Stages
        # 0-5%: Init
        # 5-10%: Enhancement
        # 10-90%: Generation
        # 90-100%: Saving
        
        def update_job_progress(percent: int, message: str = None):
            if job_id in active_jobs:
                active_jobs[job_id]["progress"] = percent
                if message:
                    active_jobs[job_id]["status_message"] = message

        def cancellation_check(step_idx, total_steps, *args):
            # Map generation progress (0-100) to Global Progress (10-90)
            gen_percent = (step_idx + 1) / total_steps
            global_percent = 10 + int(gen_percent * 80)
            
            update_job_progress(global_percent, f"Generating ({step_idx + 1}/{total_steps})")
                
            if active_jobs.get(job_id, {}).get("cancelled", False):
                raise RuntimeError(f"Job {job_id} cancelled by user.")
        
        # loop moved to start of function
        
        def _run_pipeline():
            nonlocal input_images
            result = None
            if pipeline_type == "ti2vid":
                # Use local variable to avoid UnboundLocalError with closure capture
                run_prompt = prompt
                if enhance_prompt:
                    update_job_progress(5, "Enhancing Prompt...")
                    logger.info("Enhancing prompt with Gemma...")
                    try: 
                        text_encoder = pipeline.stage_1_model_ledger.text_encoder()
                        is_image_mode = (num_frames == 1)
                        
                        # Detect if we are extending a video (Input Image + Video Output)
                        # Use Director logic
                        sys_prompt = None
                        if not is_image_mode and input_images:
                             sys_prompt = (
                                "You are an expert Prompt Engineer for the LTX-2 Video Generation model. "
                                "Your task is to take the user's Global Goal and expand it into a rich, detailed, "
                                "cinematic prompt. "
                                "CRITICAL: The output video MUST transition seamlessly from the provided input image. "
                                "Ensure visual coherence of: characters, style, lighting, environment, camera angle. "
                                "Output ONLY the prompt paragraph."
                             )
                             # Prefix user prompt effectively
                             enhancement_input = f"Global Goal: {prompt}"

                        # Logic to prevent "Character Sheet" description contamination:
                        # If the input image is inferred from Element Visuals (Reference), 
                        # DO NOT pass it to Gemma for captioning. We only want 'Action' enhancement.
                        
                        image_for_gemma = None
                        if input_images and not is_inferred_start_frame:
                            image_for_gemma = input_images[0][0]

                        run_prompt = generate_enhanced_prompt(
                            text_encoder, 
                            enhancement_input if sys_prompt else prompt, 
                            image_path=image_for_gemma,
                            seed=seed,
                            is_image=is_image_mode,
                            system_prompt=sys_prompt
                        )
                        # Save enhanced prompt immediately
                        update_job_db(job_id, "processing", enhanced_prompt=run_prompt)
                        
                        del text_encoder
                        cleanup_memory()
                    except Exception as e:
                        logger.warning(f"Prompt enhancement failed: {e}")

                # VALIDATE INPUT IMAGES (Prevent Black Output)
                if input_images:
                    valid_inputs = []
                    for item in input_images:
                        try:
                            # Item structure: (path, idx, strength)
                            start_path = item[0]
                            if os.path.exists(start_path):
                                # Fix: Pipeline expects PATHS (strings), not PIL Objects.
                                # LTX-2 media_io.py calls Image.open(path), which fails if we pass an object.
                                # So we just validate existence here.
                                valid_inputs.append(item)
                            else:
                                logger.warning(f"Skipping Missing Start Frame: {start_path}")
                        except Exception as e:
                            logger.warning(f"Failed to validate start frame {item}: {e}")
                    input_images = valid_inputs
                    
                    if not input_images:
                         logger.warning("No valid input images remaining after validation. Proceeding with T2V (No Visual Conditioning).")
                
                # Update progress before heavy model load/start
                update_job_progress(10, "Starting Generation...")
                
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
                    callback_on_step_end=cancellation_check
                )
            elif pipeline_type == "ic_lora":
                result = pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(params.get("fps", 25.0)),
                    images=input_images,
                    video_conditioning=video_cond,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                    callback_on_step_end=cancellation_check
                )
            elif pipeline_type == "keyframe":
                result = pipeline(
                    prompt=prompt,
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
                    enhance_prompt=enhance_prompt,
                    callback_on_step_end=cancellation_check
                )
            
            # Handle different return types
            if isinstance(result, tuple):
                logger.info(f"Pipeline returned tuple with {len(result)} elements")
                # Extract only first 2 values (video, audio) even if tuple has more
                if len(result) >= 2:
                    return (result[0], result[1])
                elif len(result) == 1:
                    return (result[0], None)
                else:
                    return (None, None)
            else:
                logger.info(f"Pipeline returned single value, type: {type(result)}")
                return (result, None)  # Standardize to (video, None)
        
        # Execute pipeline and unpack result
        # Note: 90% update is handled AFTER generation now to avoid premature jump.
        pipeline_result = await loop.run_in_executor(None, _run_pipeline)
        update_job_progress(90, "Finalizing & Saving...")
        logger.info(f"_run_pipeline returned: type={type(pipeline_result)}, len={len(pipeline_result) if isinstance(pipeline_result, tuple) else 'N/A'}")
        
        video, audio = pipeline_result
        
        # Get project-scoped output paths
        paths = get_project_output_paths(job_id, project_id)
        
        # Ensure output directories exist
        os.makedirs(paths["output_dir"], exist_ok=True)
        os.makedirs(paths["thumbnail_dir"], exist_ok=True)
        
        # Determine output path based on image/video mode
        if num_frames == 1:
            output_path = paths["output_path"].replace(".mp4", ".jpg")
        else:
            output_path = paths["output_path"]
        
        # Check for image mode
        if num_frames == 1:
            logger.info(f"Saving single frame as image to {output_path}")
            
            # extract first frame
            # video is (channels, frames, height, width) or (frames, channels, height, width)?
            # output of pipeline is usually (1, C, F, H, W) or (C, F, H, W) or (F, C, H, W)
            # Checked LTX pipeline: returns (C, F, H, W) usually? Or (F, C, H, W)?
            # media_io.encode_video expects:
            # "video: torch.Tensor | Iterator[torch.Tensor]"
            # "first_chunk = next(video)"
            # "_, height, width, _ = first_chunk.shape" where shape is (F, H, W, C)?
            # Actually LTX `TI2VidTwoStagesPipeline` returns `(video, audio)`
            # The tensor returned by pipeline is usually (B, C, F, H, W) or unbatched (C, F, H, W)
            # Wait, `ltx_pipelines/utils/media_io.py` `encode_video` takes `video`
            # and does `_, height, width, _ = first_chunk.shape`.
            
            # Let's look at `encode_video` implementation again in my brain.
            # `resize_aspect_ratio_preserving` rearranges to `b c f h w -> b f h w c`
            # The pipeline usually returns tensor in `b c f h w` format (0-1 range) or similar.
            
            # Let's assume `video` is the tensor returned by pipeline.
            # If `media_io.encode_video` handles it, it must be compatible.
            # `encode_video` usually takes `[C, F, H, W]` or similar and converts to `[F, H, W, C]`?
            # Actually, `media_io.py` line 216: `video_chunk_cpu = video_chunk.to("cpu").numpy()`
            # It expects `video` to be an iterator of chunks or a single tensor.
            # If it is a tensor, line 191 makes it an iterator `video = iter([video])`.
            
            # In `media_io.py`:
            # `encode_video` iterates chunks.
            # The chunks are expected to be in format ready for `av.VideoFrame.from_ndarray(..., format='rgb24')`?
            # No, line 218: `frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")`
            # This implies `frame_array` is (H, W, 3).
            # So `video_chunk_cpu` must be (F, H, W, 3).
            
            # So the pipeline output `video` must be a tensor that `encode_video` can iterate and get (F, H, W, 3) chunks?
            # Wait, `TI2VidTwoStagesPipeline` output might be different. 
            # `ltx_pipelines/ti2vid_two_stages.py`... 
            # I don't have that file open, but `worker.py` imports `encode_video` from `ltx_pipelines.utils.media_io`.
            # If `worker.py` calls `encode_video(video=video, ...)` then `video` matches `encode_video` input.
            
            # Let's just grab the tensor and convert it manually if it's 1 frame.
            
            # If `video` is the thing passed to `encode_video`, let's peek at it.
            # If it's a tensor, we can convert it.
            
            # Assuming `video` is a tensor on CPU or GPU.
            # If it works for `encode_video`, it must be convertible to numpy.
            
            # Let's safely extract the frame.
            try:
                # 1. Ensure it's on CPU
                if isinstance(video, torch.Tensor):
                    frame_tensor = video.detach().cpu()
                else:
                    # It's a generator or iterator (standard for LTX pipelines to save memory)
                    try:
                        # Consume the first chunk from the generator
                        video_iter = iter(video)
                        first_chunk = next(video_iter)
                        frame_tensor = first_chunk.detach().cpu()
                        
                        # Ensure we consume/close the generator if needed, though for 1 frame it might be done.
                        # However, if the pipeline is still running (streaming), breaking early is fine.
                    except StopIteration:
                        raise ValueError("Pipeline returned empty output for image generation.")

                # frame_tensor shape check
                logger.info(f"Extracted frame tensor shape: {frame_tensor.shape}")
                
                # Let's try to infer shape. 
                # Dimensions are usually (Batch, Channels, Frames, Height, Width) or (Channels, Frames, Height, Width).
                # `resize_aspect_ratio_preserving` returns `b f h w c`. 
                # If pipeline returns `b f h w c` (normalized 0-1 or -1..1?)
                
                # NOTE: I am making a change to `worker.py`, not the pipeline. 
                # `worker.py` receives `video` from `_run_pipeline`.
                
                # Safest way: Just inspect the tensor shape at runtime or use a robust conversion.
                
                # Check dimensions
                # If 5 dims: (B, C, F, H, W) or (B, F, H, W, C)
                # If 4 dims: (C, F, H, W) or (F, H, W, C)
                
                # Let's assume standard PyTorch image layout (C, H, W) or video (C, F, H, W).
                # LTX-2 usually works with (B, C, F, H, W).
                
                if frame_tensor.ndim == 5:
                    frame_tensor = frame_tensor.squeeze(0) # Remove batch
                
                if frame_tensor.ndim == 4:
                    # (C, F, H, W) or (F, H, W, C) or (F, C, H, W)?
                    # If C is 3, we can check.
                    if frame_tensor.shape[0] == 3: # (C, F, H, W)
                        frame_tensor = frame_tensor.permute(1, 2, 3, 0) # -> (F, H, W, C)
                    elif frame_tensor.shape[1] == 3: # (F, C, H, W)
                         frame_tensor = frame_tensor.permute(0, 2, 3, 1) # -> (F, H, W, C)
                    # If shape[-1] is 3, it's (F, H, W, C) already.
                
                # Now we have (F, H, W, C) with F=1.
                frame_tensor = frame_tensor.squeeze(0) # (H, W, C)
                
                # Check range. If float, is it -1..1 or 0..1?
                # `media_io` normalize_latent does (x / 127.5 - 1.0).
                # If pipeline returns un-normalized (0-1), then just * 255.
                # If pipeline returns -1..1, then ((x + 1) / 2) * 255.
                
                # `encode_video` takes `video`. 
                # Let's look at `media_io.py`.
                # line 218: `av.VideoFrame.from_ndarray(frame_array, format="rgb24")`
                # `frame_array` is from `video_chunk_cpu`.
                # `video_chunk_cpu` is `video_chunk.to("cpu").numpy()`.
                # If `frame_array` is uint8, it works.
                # Does `_run_pipeline` return uint8?
                
                # The pipeline `TI2VidTwoStagesPipeline` usually returns (0-1) float tensor.
                # In `helpers.py`, or `pipelines`, the decoding from VAE usually gives floats.
                # `encode_video` doesn't seem to do scaling!
                # Wait, `media_io.py`:
                # `encode_video`:
                # `frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")`
                # This requires `frame_array` to be uint8 and valid shape.
                # So the `video` tensor passed to `encode_video` MUST ALREADY BE uint8 0-255 (B, F, H, W, C).
                
                # So I can just take it, ensure it's uint8, and save.
                
                frame_arr = frame_tensor.numpy()
                
                # Robustness check: If pipeline returns floats (0..1), scale to 255
                if frame_arr.dtype != np.uint8:
                    if frame_arr.max() <= 1.0:
                         logger.info("Scaling image tensor from 0..1 to 0..255")
                         frame_arr = (frame_arr * 255).clip(0, 255)
                    frame_arr = frame_arr.astype(np.uint8)
                
                img = Image.fromarray(frame_arr)
                img.save(output_path, quality=95)
                logger.info(f"Image saved to {output_path}")
                
                # Generate a thumbnail for the image (copy or resize it)
                thumbnail_web_path = None
                try:
                    if output_path != thumb_path:
                        img.thumbnail((256, 256))
                        img.save(thumb_path, quality=85)
                        logger.info(f"Generated thumbnail: {thumb_path}")
                    # Use project-scoped URL
                    if project_id:
                        thumbnail_web_path = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
                    else:
                        thumbnail_web_path = f"/generated/{os.path.basename(thumb_path)}"
                except Exception as e:
                    logger.warning(f"Thumbnail gen failed for image: {e}")

                # Use project-scoped paths for DB
                if project_id:
                    output_url = f"/projects/{project_id}/generated/{os.path.basename(output_path)}"
                else:
                    output_url = f"/generated/{os.path.basename(output_path)}"
                
                update_job_db(
                    job_id, 
                    "completed", 
                    output_path=output_url,
                    thumbnail_path=thumbnail_web_path,
                    actual_frames=1
                )

            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                # Fallback to video just in case? 
                # No, user wants image.
                raise e
        else:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
            # For video: encode normally
            # encode_video signature: (video, fps, audio, audio_sample_rate, output_path, video_chunks_number)
            encode_video(
                video,                                                           # 1st: video tensor
                int(params.get("fps", 25)),                                     # 2nd: fps
                audio,                                                          # 3rd: audio tensor
                AUDIO_SAMPLE_RATE if audio is not None else None,              # 4th: audio_sample_rate
                output_path,                                                    # 5th: output_path
                video_chunks_number                                             # 6th: video_chunks_number
            )
            logger.info(f"Video saved to {output_path}")
        
        # Generate thumbnail (using project-scoped path)
        thumb_path = paths["thumbnail_path"]
        thumbnail_web_path = None
        try:
            if num_frames == 1:
                # For images, resize the saved image
                img = Image.open(output_path)
                img.thumbnail((256, 256))
                img.save(thumb_path, quality=85)
                logger.info(f"Generated thumbnail: {thumb_path}")
            else:
                # For videos, extract a frame using ffmpeg
                # Extract at 0.5s or 0s
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"Generated thumbnail: {thumb_path}")
            # Use project-scoped URL
            if project_id:
                thumbnail_web_path = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
            else:
                thumbnail_web_path = f"/generated/{os.path.basename(thumb_path)}"  # Legacy fallback
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")

        # Detect actual frame count to sync UI
        actual_frames = num_frames
        if num_frames > 1: # Only probe video files
            try:
                # Probe the file
                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", output_path]
                res = subprocess.check_output(cmd).decode("utf-8").strip()
                if res and res.isdigit():
                    actual_frames = int(res)
                    logger.info(f"Verified actual frame count: {actual_frames}")
                    if job_id in active_jobs:
                        active_jobs[job_id]["actual_frames"] = actual_frames
            except Exception as e:
                logger.warning(f"Could not verify frame count: {e}")

            # GENERATE THUMBNAIL (Standard)
            # GENERATE THUMBNAIL (Standard)
            thumbnail_web_path = None
            try:
                if project_id:
                    # Use thumbnails/ directory for project-based jobs
                    thumb_dir = os.path.join(PROJECTS_DIR, project_id, "thumbnails")
                    os.makedirs(thumb_dir, exist_ok=True)
                    thumb_path = os.path.join(thumb_dir, f"{job_id}_thumb.jpg")
                    thumbnail_web_path = f"/projects/{project_id}/thumbnails/{os.path.basename(thumb_path)}"
                else:
                    # Legacy fallback (same folder as video)
                    thumb_path = output_path.replace(".mp4", "_thumb.jpg")
                    thumbnail_web_path = f"/generated/{os.path.basename(thumb_path)}"

                # Extract at 0.5s
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path, "-ss", "00:00:00.500", "-vframes", "1", thumb_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"Generated thumbnail: {thumb_path}")
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail: {e}")

            # Update DB with explicit thumbnail path
            update_job_db(
                job_id, 
                "completed", 
                output_path=f"/projects/{project_id}/generated/{os.path.basename(output_path)}" if project_id else f"/generated/{os.path.basename(output_path)}",
                thumbnail_path=thumbnail_web_path,
                actual_frames=actual_frames
            )
            
            update_job_progress(100, "Done")
            
            await event_manager.broadcast("complete", {
                "job_id": job_id, 
                "url": f"/projects/{project_id}/generated/{os.path.basename(output_path)}" if project_id else f"/generated/{os.path.basename(output_path)}", 
                "type": "video",
                "thumbnail_url": thumbnail_web_path,
                "actual_frames": actual_frames
            })
            logger.info(f"Video saved to {output_path}")

async def generate_video_task(job_id: str, params: dict):
    logger.info(f"Starting generation for job {job_id}")
    active_jobs[job_id] = {
        "cancelled": False,
        "status": "processing",
        "progress": 0,
        "eta_seconds": None,
        "status_message": "Initializing..."
    }
    # Initial update
    update_job_db(job_id, "processing")
    
    try:
        pipeline_type = params.get("pipeline_type", "ti2vid")
        timeline = params.get("timeline", [])
        
        # DEBUG: Log raw params
        import json
        with open("server_last_request.json", "w") as f:
            json.dump(params, f, indent=2)
        
        # --- Dynamic Pipeline Selection & Parameter Conversion ---
        
        def resolve_path(p: str):
            """
            Resolves a path that might be a URL to a local filesystem path.
            e.g. http://localhost:8000/uploads/file.png -> /abs/path/to/uploads/file.png
                 /uploads/file.png -> /abs/path/to/uploads/file.png
            """
            if not p:
                return p
            p = str(p)
            
            # Handle full URLs first
            if "localhost" in p or "127.0.0.1" in p:
                if "/uploads/" in p:
                    filename = p.split("/uploads/")[-1]
                    return os.path.join(os.path.dirname(__file__), "uploads", filename)
                if "/generated/" in p:
                    filename = p.split("/generated/")[-1]
                    return os.path.join(os.path.dirname(__file__), "generated", filename)
            
            # Handle relative URL paths (starting with /uploads or /generated)
            if p.startswith("/uploads/"):
                filename = p.split("/uploads/")[-1]
                return os.path.join(os.path.dirname(__file__), "uploads", filename)
                
            if p.startswith("/generated/"):
                filename = p.split("/generated/")[-1]
                return os.path.join(os.path.dirname(__file__), "generated", filename)
                
            return p

        # If "advanced" mode, we infer the actual pipeline from the timeline assets
        if pipeline_type == "advanced":
            input_images = []
            video_cond = []
            
            has_video = any(t.get("type") == "video" for t in timeline)
            
            for item in timeline:
                raw_path = item.get("path")
                path = resolve_path(raw_path) # RESOLVE PATH HERE
                
                idx = item.get("frame_index", 0)
                strength = item.get("strength", 1.0)
                
                if item.get("type") == "image":
                    # (path, frame_index, strength)
                    input_images.append((path, idx, strength))
                elif item.get("type") == "video":
                    # (path, strength) - frame_index ignored for global video style transfer
                    video_cond.append((path, strength))
            
            # Logic Table:
            # 1. Video Present -> ICLora (V2V)
            # 2. Start AND End Images Present -> Keyframe Interpolation
            # 3. Else -> TI2Vid (T2V, T2I, I2V, I2I)
            
            if has_video:
                pipeline_type = "ic_lora"
                params["video_conditioning"] = video_cond
                params["images"] = input_images # ICLora can also take images? Usually just video.
            elif len(input_images) >= 2 and any(img[1] == 0 for img in input_images) and any(img[1] == (params.get("num_frames", 121)-1) for img in input_images):
                # Heuristic: If we have an image at start (0) and end (num_frames-1), assume Keyframe Interpolation
                pipeline_type = "keyframe"
                params["images"] = input_images
            else:
                pipeline_type = "ti2vid"
                params["images"] = input_images
            
            logger.info(f"Pipeline: {pipeline_type}, Input Images: {len(input_images)}, Video Cond: {len(video_cond)}")

        # Explicitly handle Image Generation (num_frames == 1)
        # If user wants an image, we should ensure we use a pipeline that supports it (ti2vid is best for T2I/I2I)
        if params.get("num_frames") == 1:
             # If automatic or something else was selected but we just want a single frame, force ti2vid
             # unless we have specific logic for others. ti2vid handles standard diffusers pipelines well.
             if pipeline_type == "auto" or pipeline_type == "advanced":
                 pipeline_type = "ti2vid"
             # If it was "keyframe" or "ic_lora" but num_frames is 1, that might be invalid?
             # IC-LoRA is video-to-video, so 1 frame might be weird.
             # Keyframe needs at least 2 frames.
             # So safely force ti2vid for single image generation.
             if pipeline_type in ["keyframe", "ic_lora"]:
                 params["pipeline_type"] = "ti2vid" # Override for worker
                 pipeline_type = "ti2vid"

        # Load the selected pipeline
        pipeline = await manager.load_pipeline(pipeline_type)
        
        # Adjust params for specific pipelines
        params["pipeline_type"] = pipeline_type
        
        # Decide if chained or standard
        num_frames = params.get("num_frames", 121)
        
        if pipeline_type == "ti2vid" and num_frames > 121:
             await generate_chained_video_task(job_id, params, pipeline)
        else:
             await generate_standard_video_task(job_id, params, pipeline)
            
    except RuntimeError as e:
        if "cancelled" in str(e):
            logger.info(f"Job {job_id} was successfully cancelled.")
            update_job_db(job_id, "cancelled")
        else:
             logger.error(f"Generation failed: {e}")
             update_job_db(job_id, "failed", error=str(e))
             import traceback
             traceback.print_exc()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        update_job_db(job_id, "failed", error=str(e))
        import traceback
        with open("server_last_error.txt", "w") as f:
            f.write(f"Error: {e}\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
    finally:
        if job_id in active_jobs:
            del active_jobs[job_id]

async def generate_image_task(job_id: str, params: dict):
    """
    Handles Flux 2 Image Generation asynchronously.
    """
    logger.info(f"Starting Image Generation for job {job_id}")
    
    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    seed = params.get("seed", 42)
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("num_inference_steps", 25)
    guidance = params.get("guidance_scale", 3.5)
    project_id = params.get("project_id")
    ip_adapter_images = params.get("reference_images", [])

    # Get paths
    if project_id:
        output_dir = os.path.join(PROJECTS_DIR, project_id, "generated", "images")
        web_prefix = f"/projects/{project_id}/generated/images"
    else:
        output_dir = os.path.join(GENERATED_DIR, "images") 
        web_prefix = "/generated/images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{job_id}.jpg" # Images usually JPG/PNG
    output_path = os.path.join(output_dir, output_filename)
    
    # Progress Callback
    # Flux callback returns (step_idx, total_steps)
    
    # Resolve Elements / IP-Adapter
    # params['reference_images'] contains element_ids
    from database import Session, engine, Element
    from sqlmodel import select
    
    resolved_ip_paths = set() # Use set for deduplication
    resolved_triggers_injected = []
    
    logger.info(f"[generate_image_task] Resolving elements. Explicit IDs: {ip_adapter_images}, Project: {project_id}")
    
    with Session(engine) as session:
        # 1. Resolve Explicitly Selected IDs (from UI dropdown)
        if ip_adapter_images:
            for el_id in ip_adapter_images:
                el = session.get(Element, el_id)
                if el:
                    resolved_path = resolve_element_image_path(el.image_path)
                    if resolved_path:
                        resolved_ip_paths.add(resolved_path)
                        logger.info(f"Resolved element {el.name}: {resolved_path}")
                    # Inject trigger if missing
                    if el.trigger_word and el.trigger_word not in prompt:
                         resolved_triggers_injected.append(el.trigger_word)
        
        # 2. Scan Prompt for Implicit Triggers (e.g. "@Hero")
        # Only if we have a project context (which we do)
        if project_id:
            all_elements = session.exec(select(Element).where(Element.project_id == project_id)).all()
            for el in all_elements:
                if el.trigger_word and el.trigger_word in prompt:
                    resolved_path = resolve_element_image_path(el.image_path)
                    if resolved_path:
                        logger.info(f"Auto-detected trigger in prompt: {el.trigger_word} -> {resolved_path}")
                        resolved_ip_paths.add(resolved_path)

    # Convert back to list for Flux
    final_ip_images = list(resolved_ip_paths) if resolved_ip_paths else None
    
    # Inject Detected Triggers (only for explicit ones that were missing)
    if resolved_triggers_injected:
        prompt = f"{prompt} {' '.join(resolved_triggers_injected)}"
        logger.info(f"Injected Explicit Triggers: {resolved_triggers_injected}. New Prompt: {prompt}")
    
    logger.info(f"[generate_image_task] Resolved IP-Adapter paths: {final_ip_images}")
    # Define callback
    def progress_callback(step, total):
        if job_id in active_jobs:
            pct = int((step / total) * 100)
            active_jobs[job_id]["progress"] = pct
            active_jobs[job_id]["status_message"] = f"Denoising step {step}/{total}"
            
            # Broadcast throttled? broadcast_progress is async, but we are in sync callback?
            # We can't await here easily unless we queue it or use run_coroutine_threadsafe.
            # But simpler: just update active_jobs and let a poller handle it? 
            # OR - since we run in executor, we can't easily call async.
            # However, broadcast_log uses event_manager.broadcast which is async.
            # Ideally we just update the in-memory state and the UI polls / WebSocket broadcasts appropriately.
            # But our event_manager is async.
            pass

    # Initialize Job State
    active_jobs[job_id] = {
        "progress": 0, 
        "status": "processing",
        "status_message": "Initializing ...", 
        "eta_seconds": steps * 1.5 # Rough guess
    }
    
    try:
        from models.flux_wrapper import flux_inpainter
        
        loop = asyncio.get_running_loop()
        
        # We need to wrap the callback to be thread-safe for asyncio broadcast if we wanted real-time push events from thread.
        # But for now `active_jobs` update is enough if we have a poller. 
        # Wait, the frontend polls /jobs/{id} right? No, /jobs/{id} logic needs to exist.
        # Currently `server.py` doesn't seem to have a generic `GET /jobs/{id}`? 
        # Existing logic uses `event_manager` (SSE).
        # To make SSE work from threaded execution, we need `loop.call_soon_threadsafe`.

        def thread_safe_callback(step, total):
            pct = int((step / total) * 100)
            msg = f"Denoising step {step}/{total}"
            
            # Update local state
            if job_id in active_jobs:
                # CHECK CANCELLATION
                if active_jobs[job_id].get("status") == "cancelling":
                    raise Exception("Job Cancelled by User")
                    
                active_jobs[job_id]["progress"] = pct
                active_jobs[job_id]["status_message"] = msg
            
            # Schedule async broadcast
            async def send_update():
               await broadcast_progress(job_id, pct, "processing")
            
            asyncio.run_coroutine_threadsafe(send_update(), loop)

        # Run Blocking Generation in Executor
        image = await loop.run_in_executor(
            None,
            lambda: flux_inpainter.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_inference_steps=steps,
                seed=seed,
                ip_adapter_images=final_ip_images,
                callback=thread_safe_callback
            )
        )
        
        # Save Image
        image.save(output_path, quality=95)
        
        # Create Asset Record
        from database import Asset, Session, engine
        asset_id = uuid.uuid4().hex
        
        with Session(engine) as session:
            asset = Asset(
                id=asset_id,
                project_id=project_id,
                type="image",
                path=output_path,
                url=f"{web_prefix}/{output_filename}",
                filename=output_filename,
                width=width,
                height=height,
                created_at=datetime.now(timezone.utc),
                meta_json=json.dumps({
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "steps": steps,
                    "guidance": guidance,
                    "reference_elements": ip_adapter_images # Store original IDs for UI restoration
                })
            )
            session.add(asset)
            session.commit()
            
        # Update Job
        update_job_db(
            job_id, 
            "completed", 
            output_path=f"{web_prefix}/{output_filename}",
            status_message="Done"
        )
        
        # Final Broadcast
        await event_manager.broadcast("complete", {
            "job_id": job_id,
            "url": f"{web_prefix}/{output_filename}",
            "type": "image",
            "asset_id": asset_id
        })
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        update_job_db(job_id, "failed", error=str(e))
        await event_manager.broadcast("error", {"job_id": job_id, "message": str(e)})
        raise e
    finally:
        if job_id in active_jobs:
            del active_jobs[job_id]
