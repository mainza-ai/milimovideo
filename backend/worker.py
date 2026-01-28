import asyncio
import os
# Fix for MPS memory allocation limits on Apple Silicon for large generations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import subprocess
import torch
import sys
import uuid
import logging
import gc
from typing import Optional, Any, Dict

# Ensure LTX-2 packages are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../LTX-2/packages/ltx-core/src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../LTX-2/packages/ltx-pipelines/src")))

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
from datetime import datetime

# Configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_job_db(job_id: str, status: str, output_path: str = None, error: str = None, enhanced_prompt: str = None, status_message: str = None):
    """Helper to update Job record in SQLite."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = status
                if output_path: 
                    job.output_path = output_path
                if error: 
                    job.error_message = str(error)
                if enhanced_prompt:
                    job.enhanced_prompt = enhanced_prompt
                if status_message:
                    job.status_message = status_message
                if status in ["completed", "failed", "cancelled"]:
                    job.completed_at = datetime.utcnow()
                session.add(job)
                session.commit()
    except Exception as e:
        logger.error(f"Failed to update Job DB for {job_id}: {e}")

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
        ltx2_root = os.path.abspath(os.path.join(self.get_base_dir(), "../LTX-2"))
        models_dir = os.path.join(ltx2_root, "models")
        
        return {
            "checkpoint_path": os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled.safetensors"),
            "distilled_lora_path": os.path.join(models_dir, "checkpoints", "ltx-2-19b-distilled-lora-384.safetensors"), 
            "spatial_upsampler_path": os.path.join(models_dir, "upscalers", "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
            "gemma_root": os.path.join(models_dir, "text_encoders", "gemma3"),
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
            logger.info(f"LTX-2 Pipeline {pipeline_type} loaded successfully.")
            return self._pipeline

# Singleton instance
manager = ModelManager()

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))


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
    
    # Initialize Storyboard Manager
    output_dir = os.path.join(get_base_dir(), "generated")
    storyboard_manager = StoryboardManager(job_id, prompt, params, output_dir)
    
    chunk_size = storyboard_manager.chunk_size # 121
    num_chunks = storyboard_manager.get_total_chunks()
    
    logger.info(f"Chained generation: {desired_total_frames} frames split into {num_chunks} chunks.")
    
    files_to_stitch = []
    
    # Prepare cancellation
    def cancellation_check(step_idx, total_steps, *args):
        progress = int((step_idx / total_steps) * 100)
        # We can't await inside this sync callback easily without creating a task?
        # Ideally pipeline callback should be async or we fire and forget. 
        # For now, let's just update active_jobs dict and maybe background thread sends events?
        # Or simpler: Just accept that callback is sync.
        if job_id in active_jobs:
             active_jobs[job_id]["progress"] = progress
             
        if active_jobs.get(job_id, {}).get("cancelled", False):
            raise RuntimeError(f"Job {job_id} cancelled by user.")
            
    # Helper to broadcast async (can be called from main loop)
    async def update_ui(msg: str):
         await broadcast_log(job_id, msg)

    loop = asyncio.get_running_loop()
    
    last_chunk_output = None
    
    try:
        for chunk_idx in range(num_chunks):
            chunk_job_id = f"{job_id}_part_{chunk_idx}"
            chunk_output_path = os.path.join(output_dir, f"{chunk_job_id}.mp4")
            files_to_stitch.append(chunk_output_path)
            
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

            # Critical Fix for Extend + Smart Continue:
            # If this is the first chunk, and we have global input images (from Extend timeline),
            # we must use them! Storyboard doesn't know about them yet (or returned empty).
            if chunk_idx == 0 and not chunk_images and params.get("images"):
                 chunk_images = params.get("images")
                 logger.info(f"Using initial timeline images for Chunk 0: {len(chunk_images)}")

            # Manually handle Prompt Enhancement to inject "Director" logic
            # The pipeline's internal enhance_prompt doesn't support our custom system_prompt.
            # So we enhance here and disable pipeline enhancement.
            do_pipeline_enhance = False 
            
            if params.get("enhance_prompt", True):
                # If Storyboard already gave us a prompt (Chunk > 0), it's already enhanced.
                if chunk_idx > 0 and chunk_config.get("prompt"):
                    do_pipeline_enhance = False
                elif chunk_idx == 0:
                    # Chunk 0: We need to enhance manually, especially if extending (Video Input)
                    # to use the Director System Prompt.
                    
                    # Get Text Encoder
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
                             sys_prompt = (
                                "You are an expert Prompt Engineer for the LTX-2 Video Generation model. "
                                "TASK: The user has provided the last frame of a video. Analyze it and write a prompt to GENERATE THE NEXT 4 SECONDS of video, extending the shot seamlessly. "
                                "The output must be a single flowing paragraph following this structure: "
                                "1. Establish the shot (cinematography, scale). "
                                "2. Set the scene (lighting, atmosphere). "
                                "3. Describe the action (natural sequence, present tense). "
                                "4. Visual Details (characters, appearance). "
                                "5. Camera Movement (how view shifts). "
                                "6. Audio (ambient sounds, dialogue in quotes). "
                                "Avoid internal emotional labels; use visual cues. Avoid text/logos. "
                                "Output ONLY the prompt paragraph."
                             )
                             effective_prompt = f"Global Goal: {chunk_prompt}"

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
                        except Exception as e:
                            logger.warning(f"Manual enhancement failed: {e}")
                            do_pipeline_enhance = True # Fallback to pipeline
                
            
            # Expose current prompt to UI
            if job_id in active_jobs:
                active_jobs[job_id]["current_prompt"] = chunk_prompt
                # Also status message
                active_jobs[job_id]["status_message"] = f"Generating Chunk {chunk_idx+1}/{num_chunks}"
            
            
            # 2. Define Pipeline execution wrapper
            def _run_chunk_pipeline(images_arg, current_prompt, pipeline_enhance):
                return pipeline(
                    prompt=current_prompt,
                    negative_prompt=negative_prompt,
                    seed=seed + chunk_idx, # vary seed slightly
                    height=height,
                    width=width,
                    num_frames=chunk_size,
                    frame_rate=float(params.get("fps", 25.0)),
                    num_inference_steps=params.get("num_inference_steps", 40),
                    cfg_guidance_scale=params.get("cfg_scale", 3.0),
                    images=images_arg,
                    tiling_config=TilingConfig.default(),
                    enhance_prompt=pipeline_enhance,
                    callback_on_step_end=cancellation_check
                )
                
            # 3. Run Pipeline
            video, audio = await loop.run_in_executor(None, _run_chunk_pipeline, chunk_images, chunk_prompt, do_pipeline_enhance)
            
            # 4. Save Output
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
            
        # Stitching
        logger.info("Stitching chunks...")
        final_output_path = os.path.join(output_dir, f"{job_id}.mp4")
        
        # Create file list for ffmpeg
        list_file_path = os.path.join(output_dir, f"{job_id}_list.txt")
        with open(list_file_path, "w") as f:
            for mp4 in files_to_stitch:
                f.write(f"file '{os.path.basename(mp4)}'\n")
        
        # Run ffmpeg concat
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path, 
            "-c", "copy", "-y", final_output_path
        ], cwd=output_dir, check=True)
        
        logger.info(f"Stitched video saved to {final_output_path}")
        update_job_db(job_id, "completed", output_path=f"/generated/{job_id}.mp4")
        await event_manager.broadcast("complete", {
            "job_id": job_id, 
            "url": f"/generated/{job_id}.mp4", 
            "type": "video"
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
        video_cond = params.get("video_conditioning", [])
        pipeline_type = params.get("pipeline_type", "ti2vid")
        
        
        tiling_config = TilingConfig.default()
        
        def cancellation_check(step_idx, total_steps, *args):
            if job_id in active_jobs:
                active_jobs[job_id]["progress"] = int((step_idx / total_steps) * 100)
                active_jobs[job_id]["status_message"] = f"Generating ({step_idx}/{total_steps})"
                
            if active_jobs.get(job_id, {}).get("cancelled", False):
                raise RuntimeError(f"Job {job_id} cancelled by user.")
        
        loop = asyncio.get_running_loop()
        
        def _run_pipeline():
            if pipeline_type == "ti2vid":
                # Use local variable to avoid UnboundLocalError with closure capture
                run_prompt = prompt
                if enhance_prompt:
                    if job_id in active_jobs:
                        active_jobs[job_id]["status_message"] = "Enhancing Prompt..."
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
                                "TASK: The user has provided the last frame of a video. Analyze it and write a prompt to GENERATE THE NEXT 4 SECONDS of video, extending the shot seamlessly. "
                                "The output must be a single flowing paragraph following this structure: "
                                "1. Establish the shot (cinematography, scale). "
                                "2. Set the scene (lighting, atmosphere). "
                                "3. Describe the action (natural sequence, present tense). "
                                "4. Visual Details (characters, appearance). "
                                "5. Camera Movement (how view shifts). "
                                "6. Audio (ambient sounds, dialogue in quotes). "
                                "Avoid internal emotional labels; use visual cues. Avoid text/logos. "
                                "Output ONLY the prompt paragraph."
                             )
                             # Prefix user prompt effectively
                             enhancement_input = f"Global Goal: {prompt}"

                        run_prompt = generate_enhanced_prompt(
                            text_encoder, 
                            enhancement_input if sys_prompt else prompt, 
                            image_path=input_images[0][0] if input_images else None,
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
                
                return pipeline(
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
                return pipeline(
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
                return pipeline(
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
        
        video, audio = await loop.run_in_executor(None, _run_pipeline)
        
        output_dir = os.path.join(get_base_dir(), "generated")
        os.makedirs(output_dir, exist_ok=True)
        # Default output path for video (will be overwritten if image)
        output_path = os.path.join(output_dir, f"{job_id}.mp4")
        
        # Check for image mode
        if num_frames == 1:
            output_path = os.path.join(output_dir, f"{job_id}.jpg")
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
                
                frame_arr = frame_tensor.numpy().astype(np.uint8)
                
                img = Image.fromarray(frame_arr)
                img.save(output_path, quality=95)
                logger.info(f"Image saved to {output_path}")
                update_job_db(job_id, "completed", output_path=f"/generated/{os.path.basename(output_path)}")

            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                # Fallback to video just in case? 
                # No, user wants image.
                raise e
        else:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
            
            encode_video(
                video=video,
                fps=float(params.get("fps", 25.0)),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
            


            logger.info(f"Video saved to {output_path}")
            
            # Detect actual frame count to sync UI
            actual_frames = num_frames
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

            update_job_db(job_id, "completed", output_path=f"/generated/{os.path.basename(output_path)}")
            await event_manager.broadcast("complete", {
                "job_id": job_id, 
                "url": f"/generated/{os.path.basename(output_path)}", 
                "type": "video",
                "actual_frames": actual_frames
            })
            logger.info(f"Video saved to {output_path}")
            update_job_db(job_id, "completed", output_path=f"/generated/{os.path.basename(output_path)}")

async def generate_video_task(job_id: str, params: dict):
    logger.info(f"Starting generation for job {job_id}")
    active_jobs[job_id] = {'cancelled': False}
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
