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
        self.chunk_size = getattr(config, "NATIVE_MAX_FRAMES", config.DEFAULT_NUM_FRAMES)
        self.overlap_frames = config.DEFAULT_OVERLAP_FRAMES
        
        # State
        self.state = StoryboardState()
        self.state.global_prompt = prompt

    def get_total_chunks(self) -> int:
        """Calculate number of chunks needed."""
        total_frames = self.params.get("num_frames", self.chunk_size)
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
            last_frame = await self._extract_last_frame(last_chunk_output, f"chunk_{chunk_idx-1}")
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
                
                enhanced, _ = llm_enhance(
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

    async def _extract_last_frame(self, video_path: str, shot_id: str) -> Optional[str]:
        """Extract last frame of video to artifacts dir using FFmpeg async to unblock event loop."""
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
            logger.info(f"Executing ffmpeg command asynchronously: {' '.join(cmd)}")
            
            import asyncio
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg failed asynchronously: {stderr.decode()}")
                return None
                
            logger.info(f"FFmpeg success. Output saved to: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to extract last frame: {e}")
            return None

    async def prepare_shot_generation(self, shot_id: str, session: Any) -> dict:
        """
        Prepares configuration for generating a specific shot.
        Returns job_params dict (prompt, images_cond, etc).
        
        Uses matched_elements from DB for element visual injection,
        falling back to trigger-word scanning if no matches are persisted.
        """
        from database import Shot, Scene
        
        shot = session.get(Shot, shot_id)
        if not shot:
            raise ValueError(f"Shot {shot_id} not found")
            
        logger.info(f"Preparing generation for Shot {shot.index} (Scene {shot.scene_id})")
        
        # 1. Base Prompt Construction
        # Start with Scene Context (Slugline)
        base_prompt = ""
        scene = None
        if shot.scene_id:
            scene = session.get(Scene, shot.scene_id)
            if scene and scene.name:
                 base_prompt += f"{scene.name}. "
             
        # Add Shot Metadata
        meta_parts = []
        if shot.shot_type:
            meta_parts.append(f"Shot Type: {shot.shot_type}")
        if shot.character:
            meta_parts.append(f"Character: {shot.character}")
        if shot.dialogue:
            meta_parts.append(f"Dialogue: {shot.dialogue}")
            
        if meta_parts:
            base_prompt += ". ".join(meta_parts) + ". "
            
        # Add Action or Prompt (Prioritize user's manual prompt)
        base_prompt += shot.prompt or shot.action or "A cinematic shot"
        
        shot_config = {
            "prompt": base_prompt,
            "images": [],
            "inspiration_images": []
        }
        
        # 2. Concept Art (Inspiration & Conditioning) Logic
        # If a thumbnail exists:
        # A) Use it as "Inspiration" (VLM Style Reference) for prompt consistency
        # B) Use it as "Input Image" (Conditioning) for video generation (Image-to-Video)
        if shot.thumbnail_url:
             from managers.element_manager import element_manager
             resolved_thumb = element_manager._resolve_element_image(shot.thumbnail_url)
             if resolved_thumb:
                 # A) Inspiration
                 shot_config["inspiration_images"].append(resolved_thumb)
                 logger.info(f"Injected Concept Art for Inspiration: {resolved_thumb}")
                 
                 # B) Conditioning (Image-to-Video)
                 # Only if no manual timeline image is set for frame 0 later
                 # We'll check this after parsing the timeline, or just prepend it now and let timeline override if needed?
                 # Safer: Prepend now. If timeline has frame 0, it should probably override or exist alongside?
                 # LTX usually takes the first match or we can filter.
                 # Let's add it now.
                 shot_config["images"].append((resolved_thumb, 0, 1.0))
                 logger.info(f"Injected Concept Art for Video Conditioning (Frame 0): {resolved_thumb}")

        # 3. Continuity Conditioning (Previous Shot in same Scene)
        prev_shot = None
        if shot.scene_id and shot.index is not None:
            logger.info(f"Querying previous shot for Scene {shot.scene_id}, Index {shot.index - 1}...")
            # Find previous shot
            prev_shot = session.exec(
                select(Shot)
                .where(Shot.scene_id == shot.scene_id)
                .where(Shot.index == shot.index - 1)
            ).first()
            logger.info(f"Previous shot query finished. Found: {prev_shot.id if prev_shot else None}")
        
        if prev_shot and prev_shot.video_url and prev_shot.status == "completed":
             # Extract last frame
             logger.info(f"Attempting to extract last frame from {prev_shot.video_url}...")
             from file_utils import resolve_path
             resolved_video_path = resolve_path(prev_shot.video_url)
             logger.info(f"Resolved video path to: {resolved_video_path}")
             
             if not resolved_video_path or not os.path.exists(resolved_video_path):
                 logger.warning(f"Resolved video path does not exist, skipping continuity extraction.")
                 last_frame_path = None
             else:
                 last_frame_path = await self._extract_last_frame(resolved_video_path, prev_shot.id)
             
             if last_frame_path:
                 # Only use continuity if we don't already have a Concept Art conditioning
                 # Priority: Manual > Concept Art > Continuity
                 has_conditioning = any(img[1] == 0 for img in shot_config["images"])
                 if not has_conditioning:
                     shot_config["images"].append((last_frame_path, 0, 1.0))
                     logger.info(f"Conditioning on Shot {prev_shot.index} last frame (Continuity).")
                 else:
                     logger.info(f"Skipping Continuity (Shot {prev_shot.index}) because Concept Art/Manual input exists.")
             else:
                 logger.warning(f"Could not extract frame from previous shot {prev_shot.id}")
        
        # 4. Element Integration — use matched_elements from DB
        from managers.element_manager import element_manager
        
        # Try matched_elements first (production path)
        if shot.matched_elements:
            import json
            try:
                matches = json.loads(shot.matched_elements)
                # Sort matches by length of trigger word (descending) to avoid partial replacements
                # e.g. @hero_face vs @hero
                matches.sort(key=lambda x: len(x.get("trigger_word", "")), reverse=True)
                
                for match in matches:
                    trigger = match.get("trigger_word", "")
                    name = match.get("element_name", "")
                    el_id = match.get("element_id")
                    
                    # Fetch full element for description
                    # We need the description to inject into the text, 
                    # because LTX-2 ignores visual embeddings (IP-Adapter).
                    from database import Element
                    element = session.get(Element, el_id)
                    description = element.description if element else ""
                    
                    # Construct Replacement: "Name (Description)"
                    replacement = name
                    if description:
                        replacement = f"{name} ({description})"
                    
                    if trigger and trigger.startswith("@") and trigger in shot_config["prompt"]:
                        # Explicit Replacement
                        shot_config["prompt"] = shot_config["prompt"].replace(trigger, replacement)
                    elif element and element.type == "location" and name not in shot_config["prompt"]:
                        # Implicit Location Injection (if not already mentioned)
                        # Append to Scene Heading part if possible, or just prepend to action
                        # Simple approach: Prepend to action part
                        # But wait, we already built base_prompt.
                        # Let's just append "Setting: Name (Description)" if it's a location match
                        shot_config["prompt"] += f". Setting: {replacement}"

                if replacement:
                     shot_config["prompt"] += f". Setting: {replacement}"

                logger.info(f"DEBUG: Prompt AFTER Element Injection: {shot_config['prompt']}")
                logger.info(f"Injected Element Descriptions into Prompt")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse matched_elements: {e}")
        
        # Fallback: trigger-word scanning (legacy path)
        # Note: We should update element_manager to use the same Name (Desc) logic
        # But for now, if matched_elements exists, we rely on that.
            
        if prev_shot:
            # Contextual Prompting (Append to end)
            context = f". Context: Following shot where {prev_shot.action}."
            shot_config["prompt"] += context

        # 5. Manual Timeline / ControlNet / IP-Adapter (from Shot.timeline)
        if shot.timeline:
            from file_utils import resolve_path
            import json
            try:
                timeline_data = json.loads(shot.timeline) if isinstance(shot.timeline, str) else shot.timeline
                if isinstance(timeline_data, list):
                    for item in timeline_data:
                        # expected item: {path, frame_index, strength, type}
                        t_path = item.get("path")
                        t_idx = item.get("frame_index", 0)
                        t_str = item.get("strength", 1.0)
                        
                        if t_path:
                            # Resolve path properly including URLs and URL encoded chars
                            resolved_timeline_path = resolve_path(t_path)
                            
                            if resolved_timeline_path:
                                # Add to images list for LTX conditioning
                                # Format: (path, frame_idx, strength)
                                shot_config["images"].append((resolved_timeline_path, t_idx, t_str))
                                logger.info(f"Injected Manual Timeline Conditioning: {resolved_timeline_path} at frame {t_idx}")
            except Exception as e:
                logger.warning(f"Failed to parse shot.timeline: {e}")

        # deduplicate frame 0 inputs?
        # If we have multiple frame 0 inputs, LTX might get confused or just use the last one?
        # Let's ensure manual timeline (user intent) overrides auto-thumbnail.
        # Current order: Thumbnail added first (Step 2), then Manual Timeline (Step 5).
        # So Manual Timeline > Thumbnail. This is correct.
        
        # Verify valid inputs
        valid_images = []
        for img in shot_config["images"]:
            path, idx, strength = img
            if os.path.exists(path):
                valid_images.append(img)
            else:
                 logger.warning(f"Skipping missing input image {path}")
        shot_config["images"] = valid_images

            
        return shot_config

    def cleanup(self):
        """Cleanup temporary files."""
        if self.artifacts_dir and os.path.exists(self.artifacts_dir):
            try:
                shutil.rmtree(self.artifacts_dir)
                logger.info(f"Cleaned up artifacts: {self.artifacts_dir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

