import os
import re
import math
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def build_complex_filtergraph(
    project_fps: int,
    project_width: int,
    project_height: int,
    v1_shots: List[Dict[str, Any]],
    v2_shots: List[Dict[str, Any]],
    a1_shots: List[Dict[str, Any]],
    output_path: str
) -> List[str]:
    """
    Builds a robust FFmpeg `-filter_complex` command for NLE timeline rendering.
    Supports resolution scaling/padding, alpha compositing (V2), and delayed audio mixing (A1).
    """

    # 1. Calculate final project duration based on the max end frame of all tracks
    max_frame = 0
    all_shots = v1_shots + v2_shots + a1_shots
    for s in all_shots:
        end_frame = s["start_frame"] + s["duration_frames"]
        if end_frame > max_frame:
            max_frame = end_frame
            
    if max_frame == 0:
        logger.warning("No clips to render! Defaulting to 1 frame.")
        max_frame = 1
        
    duration_sec = max_frame / project_fps

    # 2. Gather Inputs
    cmd = ["ffmpeg", "-y"]
    
    # Input 0: Create a solid black background canvas
    cmd.extend([
        "-f", "lavfi",
        "-i", f"color=c=black:s={project_width}x{project_height}:r={project_fps}:d={duration_sec}"
    ])
    
    input_idx = 1
    filter_chains = []
    
    # Track video/audio output stream labels
    video_outputs = []
    audio_outputs = []
    
    # 3. Process Video Clips (V1 and V2)
    def process_video_track(shots, track_label):
        nonlocal input_idx
        track_outputs = []
        
        for idx, shot in enumerate(shots):
            path = shot["path"]
            start_f = shot["start_frame"]
            trim_in_f = shot["trim_in"]
            # To ensure we don't bleed past the user's intent, calculate active duration
            duration_f = shot["duration_frames"]
            end_f = start_f + duration_f
            
            # Convert frames to seconds
            start_s = start_f / project_fps
            end_s = end_f / project_fps
            trim_in_s = trim_in_f / project_fps
            duration_s = duration_f / project_fps
            
            cmd.extend(["-i", path])
            
            # The filter chain for this specific clip
            fc = f"[{input_idx}:v]"
            
            # a) Trim
            # trim limits the clip, setpts shifts its timestamps to start at 0 internally
            fc += f"trim=start={trim_in_s}:duration={duration_s},setpts=PTS-STARTPTS"
            
            # b) Scale and Pad (Conform clip to project dimensions)
            # scale forces it to fit inside the project bounding box while maintaining aspect ratio
            # pad fills the remainder with black
            fc += f",scale={project_width}:{project_height}:force_original_aspect_ratio=decrease"
            fc += f",pad={project_width}:{project_height}:(ow-iw)/2:(oh-ih)/2"
            
            # c) Set Timeline Position
            # setpts moves the internal timeline zero-point to the project start_time
            fc += f",setpts=PTS+{start_s}/TB"
            
            out_label = f"[{track_label}_{idx}]"
            fc += out_label
            
            filter_chains.append(fc)
            track_outputs.append((out_label, start_s, end_s))
            input_idx += 1
            
        return track_outputs

    v1_processed = process_video_track(v1_shots, "v1")
    v2_processed = process_video_track(v2_shots, "v2")
    
    # 4. Process Audio Clips (A1)
    for idx, shot in enumerate(a1_shots):
        path = shot["path"]
        start_f = shot["start_frame"]
        trim_in_f = shot["trim_in"]
        duration_f = shot["duration_frames"]
        
        start_s = start_f / project_fps
        trim_in_s = trim_in_f / project_fps
        duration_s = duration_f / project_fps
        
        cmd.extend(["-i", path])
        
        fc = f"[{input_idx}:a]"
        
        # Trim audio
        fc += f"atrim=start={trim_in_s}:duration={duration_s},asetpts=PTS-STARTPTS"
        
        # Delay audio to its timeline position
        delay_ms = int(start_s * 1000)
        fc += f",adelay={delay_ms}|{delay_ms}"
        
        out_label = f"[a1_{idx}]"
        fc += out_label
        
        filter_chains.append(fc)
        audio_outputs.append(out_label)
        input_idx += 1


    # 5. Composite V1 over the base canvas
    current_bg = "[0:v]" # Start with black canvas
    if v1_processed:
        for idx, (vid_label, start_s, end_s) in enumerate(v1_processed):
            next_bg = f"[bg_v1_{idx}]"
            # overlay=enable... means this clip is only visible during its absolute timeline segment
            filter_chains.append(f"{current_bg}{vid_label}overlay=enable='between(t,{start_s},{end_s})':eof_action=pass{next_bg}")
            current_bg = next_bg
            
    # Composite V2 over the V1 master
    if v2_processed:
        for idx, (vid_label, start_s, end_s) in enumerate(v2_processed):
            next_bg = f"[bg_v2_{idx}]"
            filter_chains.append(f"{current_bg}{vid_label}overlay=enable='between(t,{start_s},{end_s})':eof_action=pass{next_bg}")
            current_bg = next_bg

    final_video_label = current_bg

    # 6. Mix Audio
    final_audio_label = None
    if audio_outputs:
        amix_inputs = "".join(audio_outputs)
        num_inputs = len(audio_outputs)
        final_audio_label = "[a_out]"
        filter_chains.append(f"{amix_inputs}amix=inputs={num_inputs}:duration=first:dropout_transition=0{final_audio_label}")
    

    # 7. Finalize Command
    cmd.extend(["-filter_complex", ";".join(filter_chains)])
    
    cmd.extend(["-map", final_video_label])
    if final_audio_label:
        cmd.extend(["-map", final_audio_label])
        
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest", # End when the shortest stream ends (bound by the black canvas)
        output_path
    ])
    
    return cmd
