import os
import logging
import shutil
import subprocess
from typing import Optional, Any
from sqlmodel import select
import config

logger = logging.getLogger(__name__)

class StoryboardState:
    def __init__(self):
        self.global_prompt = None
        self.chunks = [] # Track committed chunks

class StoryboardManager:
    def __init__(self, job_id: str = None, prompt: str = None, params: dict = None, output_dir: str = None):
        """
        Initialize Manager.
        Supports dual modes:
        1. Worker/Generation Mode: (job_id, prompt, params, output_dir)
        2. Server/Prep Mode: (output_dir=...)
        """
        self.job_id = job_id
        self.prompt = prompt
        self.params = params or {}
        self.output_dir = output_dir

        if not self.output_dir and self.params.get("workspace_dir"):
             self.output_dir = self.params.get("workspace_dir")
             
        # Ensure temp artifacts dir
        if self.output_dir:
            self.artifacts_dir = os.path.join(self.output_dir, "storyboard_artifacts")
            os.makedirs(self.artifacts_dir, exist_ok=True)
        else:
            self.artifacts_dir = None

        # Chaining Configuration — centralized in config.py
        self.chunk_size = config.DEFAULT_NUM_FRAMES
        self.overlap_frames = config.DEFAULT_OVERLAP_FRAMES
        
        # State
        self.state = StoryboardState()
        self.state.global_prompt = prompt

    def get_total_chunks(self) -> int:
        """Calculate number of chunks needed."""
        total_frames = self.params.get("num_frames", config.DEFAULT_NUM_FRAMES)
        if total_frames <= self.chunk_size:
            return 1
            
        import math
        effective_step = self.chunk_size - self.overlap_frames
        remaining = total_frames - self.chunk_size
        if remaining <= 0: return 1
        
        additional = math.ceil(remaining / effective_step)
        return 1 + additional

    async def prepare_next_chunk(self, chunk_idx, last_chunk_output, text_encoder=None) -> dict:
        """
        Prepare params for the next chunk.
        If chunk > 0, extract last frame of previous chunk and use as conditioning,
        and optionally enhance the prompt for narrative continuation.
        """
        logger.info(f"Preparing chunk {chunk_idx}...")
        
        base_prompt = self.state.global_prompt or self.prompt
        
        chunk_config = {
            "prompt": base_prompt,
            "images": [] # Conditioning
        }
        
        if chunk_idx > 0 and last_chunk_output:
            # Extract last frame for overlap conditioning
            last_frame = self._extract_last_frame(last_chunk_output, f"chunk_{chunk_idx-1}")
            if last_frame:
                 # (path, frame_idx, strength)
                 # LTX conditioning at frame 0
                 chunk_config["images"] = [(last_frame, 0, 1.0)]
                 logger.info(f"Conditioned chunk {chunk_idx} on {last_frame}")
            else:
                logger.warning(f"Failed to get condition frame for chunk {chunk_idx}")
            
            # Narrative continuation via LLM (enhance prompt for this chunk)
            try:
                from llm import enhance_prompt as llm_enhance
                
                director_prompt = (
                    "You are a Film Director continuing a video scene. "
                    "Given the Global Story Goal, write the next 4 seconds of action. "
                    "Maintain visual continuity, character identity, and narrative flow. "
                    "Specify camera movement, lighting, and atmosphere. "
                    "Output ONLY the prompt. Single paragraph."
                )
                continuation_input = (
                    f"Global Story Goal: {base_prompt}. "
                    f"This is chunk {chunk_idx + 1} of a continuous video. "
                    f"Continue the action from where the previous chunk left off."
                )
                
                enhanced = llm_enhance(
                    prompt=continuation_input,
                    system_prompt=director_prompt,
                    is_video=True,
                    text_encoder=text_encoder,
                    image_path=last_frame if last_frame else None,
                    seed=self.params.get("seed", 42),
                )
                if enhanced and enhanced != continuation_input:
                    chunk_config["prompt"] = enhanced
                    logger.info(f"Chunk {chunk_idx} enhanced prompt: {enhanced[:80]}...")
            except Exception as e:
                logger.warning(f"Chunk {chunk_idx} narrative enhancement failed: {e}")
                
        return chunk_config

    async def commit_chunk(self, chunk_idx, path, prompt):
        """Register completed chunk."""
        self.state.chunks.append({
            "index": chunk_idx,
            "path": path,
            "prompt": prompt
        })
        logger.info(f"Committed chunk {chunk_idx}: {path}")

    def _extract_last_frame(self, video_path: str, shot_id: str) -> Optional[str]:
        """Extract last frame of video to artifacts dir using FFmpeg."""
        if not self.artifacts_dir:
            return None
        try:
            filename = f"{shot_id}_last.png"
            out_path = os.path.join(self.artifacts_dir, filename)
            
            if os.path.exists(out_path):
                return out_path
                
            cmd = [
                "ffmpeg", "-y",
                "-sseof", "-0.1",  # Seek to last 0.1 sec
                "-i", video_path,
                "-vsync", "0",
                "-q:v", "2",
                "-update", "1",
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return out_path
        except Exception as e:
            logger.error(f"Failed to extract last frame: {e}")
            return None

    async def prepare_shot_generation(self, shot_id: str, session: Any) -> dict:
        """
        Prepares configuration for generating a specific shot.
        Returns job_params dict (prompt, images_cond, etc).
        """
        from database import Shot, Scene
        
        shot = session.get(Shot, shot_id)
        if not shot:
            raise ValueError(f"Shot {shot_id} not found")
            
        logger.info(f"Preparing generation for Shot {shot.index} (Scene {shot.scene_id})")
        
        # 1. Base Params
        shot_config = {
            "prompt": shot.action, # Start with base action
            "images": []
        }
        
        # 2. Conditioning (Previous Shot in same Scene)
        # Find previous shot
        prev_shot = session.exec(
            select(Shot)
            .where(Shot.scene_id == shot.scene_id)
            .where(Shot.index == shot.index - 1)
        ).first()
        
        if prev_shot and prev_shot.video_url and prev_shot.status == "completed":
             # Extract last frame
             last_frame_path = self._extract_last_frame(prev_shot.video_url, prev_shot.id)
             if last_frame_path:
                 # Add as conditioning
                 # LTX-2 conditioning format: (path, frame_idx, strength)
                 # We simply condition the START of the new shot with the END of the old one.
                 shot_config["images"] = [(last_frame_path, 0, 1.0)] 
                 logger.info(f"Conditioning on Shot {prev_shot.index} last frame.")
             else:
                 logger.warning(f"Could not extract frame from previous shot {prev_shot.id}")
        
        # 3. Prompt Enhancement (Narrative Flow)
        from managers.element_manager import element_manager
        enriched_prompt, element_visuals = element_manager.inject_elements_into_prompt(shot.action, shot.project_id)
        
        if prev_shot:
            # Contextual Prompting
            context = f"Following the previous shot where {prev_shot.action}. "
            enriched_prompt = f"{context} {enriched_prompt}"
            
        shot_config["prompt"] = enriched_prompt
        shot_config["element_images"] = element_visuals
        
        return shot_config

    def cleanup(self):
        """Cleanup temporary files."""
        if self.artifacts_dir and os.path.exists(self.artifacts_dir):
            try:
                shutil.rmtree(self.artifacts_dir)
                logger.info(f"Cleaned up artifacts: {self.artifacts_dir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

