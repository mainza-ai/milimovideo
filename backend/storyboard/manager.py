import os
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    chunk_index: int
    output_path: str
    start_frame: int
    end_frame: int
    prompt_used: str

@dataclass
class StoryboardState:
    job_id: str
    global_prompt: str
    # Context summary tracking the narrative flow
    current_context: str = ""
    # History of generated chunks
    generated_chunks: List[ChunkMetadata] = field(default_factory=list)
    # Path to the last N frames of the most recent chunk
    last_generation_frames_dir: Optional[str] = None

class StoryboardManager:
    def __init__(self, job_id: str, prompt: str, params: dict, output_dir: str):
        self.job_id = job_id
        self.params = params
        self.output_dir = output_dir
        self.state = StoryboardState(
            job_id=job_id,
            global_prompt=prompt
        )
        
        # Configuration
        # Number of frames to overlap/condition on
        self.overlap_frames = params.get("overlap_frames", 8)
        self.total_frames = params.get("num_frames", 121)
        self.chunk_size = 121 # Native model limit logic
        
        # Ensure working directory for this job's artifacts exists
        self.job_work_dir = os.path.join(output_dir, f"{job_id}_artifacts")
        os.makedirs(self.job_work_dir, exist_ok=True)

    def get_total_chunks(self) -> int:
        """Calculate number of chunks needed based on total frames and overlap."""
        # Effective new frames per chunk = chunk_size - overlap_frames
        # We need to cover (total_frames - overlap_frames) *new* content 
        # (plus the initial overlap if we count from 0, but usually first chunk is full new)
        
        # First chunk: chunk_size frames.
        # Remaining: total_frames - chunk_size.
        # Each subsequent chunk adds (chunk_size - overlap_frames) new frames.
        
        if self.total_frames <= self.chunk_size:
            return 1
            
        effective_step = self.chunk_size - self.overlap_frames
        remainder = self.total_frames - self.chunk_size
        
        num_additional = remainder // effective_step
        if remainder % effective_step != 0:
            num_additional += 1
            
        return 1 + num_additional

    async def prepare_next_chunk(self, chunk_index: int, prev_chunk_output_path: str = None, text_encoder=None) -> dict:
        """
        Prepares the configuration for the next chunk generation.
        Returns a dict of arguments to pass to the pipeline.
        
        Args:
            chunk_index: Index of the chunk to generate.
            prev_chunk_output_path: Path to the video file of the previous chunk.
            text_encoder: Optional (but recommended) text encoder model for generating enhanced prompts.
        """
        logger.info(f"StoryboardManager: Preparing chunk {chunk_index} for job {self.job_id}")
        
        chunk_config = {
            "prompt": self.state.global_prompt, # Default fallback
            "images": [] 
        }

        # 1. Conditioning Setup
        last_frame_path = None
        if chunk_index > 0 and prev_chunk_output_path:
            frames_dir = os.path.join(self.job_work_dir, f"chunk_{chunk_index-1}_end")
            os.makedirs(frames_dir, exist_ok=True)
            
            extracted_frames = self._extract_last_n_frames(
                video_path=prev_chunk_output_path,
                n=self.overlap_frames,
                output_dir=frames_dir
            )
            
            if extracted_frames:
                # Store the very last frame for prompt generation
                last_frame_path = extracted_frames[-1] 
                
                # Build images list for conditioning
                # Format: list[(path, frame_idx, strength)]
                # Strategy: Condition the FIRST 'overlap_frames' of the NEW chunk
                # with the LAST 'overlap_frames' of the OLD chunk.
                images_arg = []
                for i, frame_path in enumerate(extracted_frames):
                    images_arg.append((frame_path, i, 1.0))
                
                chunk_config["images"] = images_arg
                logger.info(f"Conditioning chunk {chunk_index} with {len(images_arg)} frames.")

        # 2. Prompt Generation (Narrative Logic)
        if chunk_index > 0 and text_encoder and last_frame_path:
             try:
                 new_prompt = await self.generate_narrative_prompt(text_encoder, last_frame_path)
                 if new_prompt:
                     chunk_config["prompt"] = new_prompt
             except Exception as e:
                 logger.warning(f"Failed to generate narrative prompt: {e}")

        # Update context for next time? (Actually we do this in commit_chunk usually)

        return chunk_config

    async def generate_narrative_prompt(self, text_encoder, image_path: str) -> str:
        """
        Uses the provided text_encoder (Gemma) to generate a narratively consistent prompt
        based on the global goal and the last frame context.
        """
        from ltx_pipelines.utils.helpers import generate_enhanced_prompt, cleanup_memory
        
        # Construct the "Director" prompt
        # We include previous context if available
        context_str = f" Previous context: {self.state.current_context}" if self.state.current_context else ""
        
        director_prompt = (
            f"You are a director. Global Story Goal: {self.state.global_prompt}.{context_str} "
            f"Task: Describe the next 4 seconds of video starting from the attached image. "
            f"Focus on continuing the action and narrative logic seamlessly."
        )
        
        logger.info(f"Generating narrative prompt with Director instruction: {director_prompt}")
        
        # Run inference (synchronously usually, but wrapped if needed)
        # generate_enhanced_prompt is blocking, so maybe we should run in executor if heavy?
        # But here we are just calling it.
        # Define the specialized Director System Prompt for LTX-2
        # Based on LTX-2 best practices: Detailed, chronological, specific lighting/camera/action.
        director_system_prompt = (
            "You are a visionary Film Director and Cinematographer. Your goal is to continue the narrative flow of a video scene.\n"
            "You will be given the last frame of the previous shot and a global story goal.\n"
            "TASK: Describe the next 4 seconds of video action. The transition must be seamless.\n"
            "GUIDELINES:\n"
            "- Analyze the visual context of the input image (characters, clothing, lighting, background).\n"
            "- Describe the ACTION that happens next. Do not just describe the static image.\n"
            "- Maintain character identity and visual consistency.\n"
            "- CINEMATOGRAPHY: Specify camera movement (e.g., 'slow dolly in', 'pan right', 'handheld', 'static').\n"
            "- LIGHTING: Describe the lighting atmosphere (e.g., 'cinematic lighting', 'soft morning light', 'neon rim light').\n"
            "- AUDIO: Include a description of the soundscape (ambient sounds, Foley, dialogue if applicable) integrated into the narrative.\n"
            "- Output ONLY the prompt for the next shot. Single paragraph, chronological flow. No polite conversation."
        )
        
        # We pass the Story Goal as the "User Prompt" so the model knows WHAT to film.
        # The system prompt tells it HOW to film it (continuation).
        user_plot_prompt = f"Global Story Goal: {self.state.global_prompt}. {context_str}"

        try:
            enhanced_prompt = generate_enhanced_prompt(
                text_encoder,
                prompt=user_plot_prompt,
                image_path=image_path,
                system_prompt=director_system_prompt
            )
            logger.info(f"Generated Narrative Prompt: {enhanced_prompt}")
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error in generate_narrative_prompt: {e}")
            return None

    async def commit_chunk(self, chunk_index: int, output_path: str, prompt_used: str):
        """
        Register a completed chunk.
        """
        # Calculate frame range (approximate for tracking)
        start = 0 
        end = self.chunk_size
        
        metadata = ChunkMetadata(
            chunk_index=chunk_index,
            output_path=output_path,
            start_frame=start,
            end_frame=end,
            prompt_used=prompt_used
        )
        self.state.generated_chunks.append(metadata)
        
        # Update Context smartly?
        # Maybe keep last 2 prompts?
        self.state.current_context = prompt_used[:200] + "..." # Truncate to avoid exploding context
        
        logger.info(f"Committed chunk {chunk_index} to Storyboard state.")

    def cleanup(self):
        """Cleanup temporary files."""
        try:
            if os.path.exists(self.job_work_dir):
                shutil.rmtree(self.job_work_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup storyboard artifacts: {e}")

    def _get_next_prompt(self, chunk_index: int) -> str:
        """
        Legacy method, superseded by generate_narrative_prompt logic in prepare_next_chunk.
        """
        return self.state.global_prompt

    def _extract_last_n_frames(self, video_path: str, n: int, output_dir: str) -> List[str]:
        """
        Extract the last n frames from a video file into output_dir.
        Returns list of absolute file paths to the images.
        """
        # We use ffmpeg.
        # To get exactly last N frames is tricky with raw ffmpeg if we don't know duration exactly.
        # But we can use -sseof (seek from end of file).
        # Assuming 25fps.
        # Duration for N frames = N / 25.0
        # Give a buffer of 2x to be safe, then slice.
        
        duration_to_extract = (n + 5) / 25.0
        
        temp_extract_dir = os.path.join(output_dir, "temp_all")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-sseof", f"-{duration_to_extract}",
            "-i", video_path,
            "-vsync", "0",
            "-q:v", "2",
            os.path.join(temp_extract_dir, "%04d.png")
        ]
        
        try:
            # Suppress output unless error
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            # List files
            files = sorted([
                os.path.join(temp_extract_dir, f) 
                for f in os.listdir(temp_extract_dir) 
                if f.endswith('.png')
            ])
            
            if not files:
                return []
                
            # Take last N
            last_n = files[-n:] if len(files) >= n else files
            
            # Move them to final output_dir with 0-indexed names
            final_files = []
            for i, src in enumerate(last_n):
                dst = os.path.join(output_dir, f"frame_{i:03d}.png")
                shutil.copy(src, dst)
                final_files.append(dst)
                
            # Clean temp
            shutil.rmtree(temp_extract_dir)
            
            return final_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e}")
            return []
