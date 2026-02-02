import os
import logging
import shutil
import subprocess
from typing import Optional, Any
from sqlmodel import select

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

        # Chaining Configuration
        self.chunk_size = 121
        self.overlap_frames = 24 
        
        # State
        self.state = StoryboardState()
        self.state.global_prompt = prompt

    def get_total_chunks(self) -> int:
        """Calculate number of chunks needed."""
        total_frames = self.params.get("num_frames", 121)
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
        If chunk > 0, extract last frame of previous chunk and use as conditioning.
        """
        logger.info(f"Preparing chunk {chunk_idx}...")
        
        base_prompt = self.state.global_prompt or self.prompt
        
        config = {
            "prompt": base_prompt,
            "images": [] # Conditioning
        }
        
        if chunk_idx > 0 and last_chunk_output:
            # Extract last frame for overlap conditioning
            last_frame = self._get_last_frame_local(last_chunk_output, f"chunk_{chunk_idx-1}")
            if last_frame:
                 # (path, frame_idx, strength)
                 # LTX conditioning at frame 0
                 config["images"] = [(last_frame, 0, 1.0)]
                 logger.info(f"Conditioned chunk {chunk_idx} on {last_frame}")
            else:
                logger.warning(f"Failed to get condition frame for chunk {chunk_idx}")
                
        # Optional: Mutate prompt based on chunk index or LLM (omitted for speed)
        return config

    async def commit_chunk(self, chunk_idx, path, prompt):
        """Register completed chunk."""
        self.state.chunks.append({
            "index": chunk_idx,
            "path": path,
            "prompt": prompt
        })
        logger.info(f"Committed chunk {chunk_idx}: {path}")

    def _get_last_frame_local(self, video_path: str, shot_id: str) -> Optional[str]:
        """Local helper to extract last frame (avoiding self._get_last_frame conflict)."""
        if not self.artifacts_dir: return None
        try:
            filename = f"{shot_id}_last.png"
            out_path = os.path.join(self.artifacts_dir, filename)
            
            if os.path.exists(out_path): return out_path
                
            cmd = [
                "ffmpeg", "-y",
                "-sseof", "-0.1", 
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
        config = {
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
             last_frame_path = self._get_last_frame(prev_shot.video_url, prev_shot.id)
             if last_frame_path:
                 # Add as conditioning
                 # LTX-2 conditioning format: (path, frame_idx, strength)
                 # We simply condition the START of the new shot with the END of the old one.
                 config["images"] = [(last_frame_path, 0, 1.0)] 
                 logger.info(f"Conditioning on Shot {prev_shot.index} last frame.")
             else:
                 logger.warning(f"Could not extract frame from previous shot {prev_shot.id}")
        
        # 3. Prompt Enhancement (Narrative Flow)
        # We can implement prompt enhancement here or let the worker do it.
        # But the plan says we do it here (or via ElementManager).
        
        from managers.element_manager import element_manager
        enriched_prompt = element_manager.inject_elements_into_prompt(shot.action, shot.project_id)
        
        if prev_shot:
            # Contextual Prompting
            # "Previous action was [prev_action]. Now, [current_action]."
            context = f"Following the previous shot where {prev_shot.action}. "
            enriched_prompt = f"{context} {enriched_prompt}"
            
        config["prompt"] = enriched_prompt
        
        return config

    def _get_last_frame(self, video_path: str, shot_id: str) -> Optional[str]:
        """Extract last frame of video to artifacts dir."""
        try:
            filename = f"{shot_id}_last.png"
            out_path = os.path.join(self.artifacts_dir, filename)
            
            if os.path.exists(out_path):
                return out_path
                
            cmd = [
                "ffmpeg", "-y",
                "-sseof", "-0.1", # Seek to last 0.1 sec
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

    def cleanup(self):
        """Cleanup temporary files."""
        # TODO: Implement granular cleanup if needed
        pass
