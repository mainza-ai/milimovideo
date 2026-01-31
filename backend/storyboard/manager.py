import os
import logging
import shutil
import subprocess
from typing import Optional, Any
from sqlmodel import select

logger = logging.getLogger(__name__)

class StoryboardManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # Ensure temp artifacts dir
        self.artifacts_dir = os.path.join(output_dir, "storyboard_artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

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
