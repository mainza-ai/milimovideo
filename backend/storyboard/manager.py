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
        # REDUCED from 8 to 4 to prevent "static lock" (pause) at joins.
        self.overlap_frames = params.get("overlap_frames", 4)
        self.total_frames = params.get("num_frames", 121)
        self.chunk_size = params.get("chunk_size", 121) 
        
        # Ensure working directory for this job's artifacts exists
        self.job_work_dir = os.path.join(output_dir, f"{job_id}_artifacts")
        os.makedirs(self.job_work_dir, exist_ok=True)

    def get_total_chunks(self) -> int:
        """Calculate number of chunks needed based on total frames and overlap."""
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
                
                images_arg = []
                count = len(extracted_frames)
                for i, frame_path in enumerate(extracted_frames):
                    if i == 0:
                        strength = 1.0
                    else:
                        # Hard Taper: 1.0 -> 0.0
                        # We must release the "static anchor" completely by the end of the overlap.
                        progress = i / max(1, count - 1)
                        strength = max(0.0, 1.0 - progress) 

                    images_arg.append((frame_path, i, strength))
                
                chunk_config["images"] = images_arg
                logger.info(f"Conditioning chunk {chunk_index} with {len(images_arg)} frames (Hard Taper 1.0->0.0).")

        # 2. Prompt Generation (Narrative Logic)
        if chunk_index > 0 and text_encoder and last_frame_path:
             try:
                 prev_prompt = None
                 if chunk_index > 0:
                     # Attempt to find chunk_index - 1
                     prev_meta = next((c for c in self.state.generated_chunks if c.chunk_index == chunk_index - 1), None)
                     if prev_meta:
                         prev_prompt = prev_meta.prompt_used

                 new_prompt = await self.generate_narrative_prompt(text_encoder, last_frame_path, previous_prompt=prev_prompt)
                 if new_prompt:
                     chunk_config["prompt"] = new_prompt
             except Exception as e:
                 logger.warning(f"Failed to generate narrative prompt: {e}")

        # Update context for next time? (Actually we do this in commit_chunk usually)

        return chunk_config

    async def generate_narrative_prompt(self, text_encoder, image_path: str, previous_prompt: str = None) -> str:
        """
        Uses the provided text_encoder (Gemma) to generate a narratively consistent prompt
        based on the global goal and the last frame context.
        """
        from ltx_pipelines.utils.helpers import generate_enhanced_prompt, cleanup_memory
        
        # EXTRACT VISUAL ANCHOR from Global Goal
        # Simple heuristic: Take the first 2 sentences which usually define the character/setting.
        # Or just pass the whole thing as "Visual Definition"
        visual_anchor = " ".join(self.state.global_prompt.split(".")[:2]) + "."
        
        director_prompt = (
            f"GLOBAL VISUAL DEFINITION (MUST MAINTAIN): {visual_anchor}\n"
            f"Current Story Goal: {self.state.global_prompt}.\n"
            f"Previous Shot Action: {previous_prompt if previous_prompt else 'None'}.\n"
            f"TASK: Write a prompt for the NEXT 4 seconds of video.\n"
            f"INSTRUCTIONS:\n"
            f"1. Start with the Visual Definition to re-anchor the model.\n"
            f"2. IMMEDIATELY describe the NEW dynamic action (Use verbs like 'runs', 'shatters', 'explodes').\n"
            f"3. Ensure the action advances the Global Story Goal.\n"
        )
        
        logger.info(f"Generating narrative prompt with Visual Anchor: {visual_anchor}")
        
        # Define the specialized Director System Prompt for LTX-2
        # Enhanced to be stricter about Goal Adherence.
        director_system_prompt = (
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
            "- AUDIO: Include a description of the soundscape (ambient sounds, Foley, dialogue if applicable).\n"
            "- Output ONLY the prompt for the next shot. Single paragraph, chronological flow. No polite conversation."
        )
        
        # We pass the formatted input as the "User Prompt"
        user_plot_prompt = director_prompt

        try:
            enhanced_prompt = generate_enhanced_prompt(
                text_encoder,
                prompt=user_plot_prompt,
                image_path=image_path,
                system_prompt=director_system_prompt
            )
            
            # --- Data Trace Logging ---
            try:
                trace_file = os.path.join(self.output_dir, "prompt_trace.txt")
                with open(trace_file, "a") as f:
                    f.write(f"\n--- Chunk Generation Trace ---\n")
                    f.write(f"Global Goal: {self.state.global_prompt}\n")
                    f.write(f"Previous Prompt: {previous_prompt}\n")
                    f.write(f"Generated Prompt: {enhanced_prompt}\n")
                    f.write(f"Context Drift Check: {'PASS' if self.state.global_prompt in director_prompt else 'FAIL'}\n") # Simple check
                    f.write("-----------------------------\n")
            except Exception as e:
                logger.warning(f"Failed to log prompt trace: {e}")
            # --------------------------

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
        # ROBUST IMPLEMENTATION:
        # 1. Get total frame count using ffprobe
        # 2. Calculate indices of last N frames
        # 3. Use ffmpeg select filter to extract exact indices
        
        try:
            # 1. Probe frame count
            # Use -count_packets to be sure, or stream info. 
            # stream=nb_read_packets is most accurate but slower. nb_frames is often missing.
            cmd_probe = [
                "ffprobe", "-v", "error", 
                "-select_streams", "v:0", 
                "-count_packets", 
                "-show_entries", "stream=nb_read_packets", 
                "-of", "csv=p=0", 
                video_path
            ]
            result = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            total_frames = int(result.stdout.strip())
            
            if total_frames <= 0:
                logger.error(f"FFprobe returned invalid frame count: {total_frames}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to probe video frame count: {e}")
            # Fallback to legacy method? No, legacy was broken. Fail hard.
            return []

        # 2. Calculate indices
        # We want frames [total-n, total-n+1, ..., total-1]
        start_idx = max(0, total_frames - n)
        # Construct select filter: "eq(n,100)+eq(n,101)+..."
        # Actually easier: "gte(n, start_idx)"
        
        temp_extract_dir = os.path.join(output_dir, "temp_all")
        os.makedirs(temp_extract_dir, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select=gte(n\\,{start_idx})",
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
                logger.warning("FFmpeg extraction produced no files.")
                return []
                
            # Take last N (in case gte produced more than N due to vsync quirks)
            last_n = files[-n:] if len(files) >= n else files
            
            # Validation: Check if files are identical (duplicate frame bug)
            # This is expensive but good for debugging. 
            # For now, trust the index-based extraction.
            
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
            logger.error(f"FFmpeg extraction failed: {e}")
            return []
