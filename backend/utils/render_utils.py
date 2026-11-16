import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def build_complex_filtergraph(
    project_fps: int,
    project_width: int,
    project_height: int,
    v1_shots: List[Dict[str, Any]],
    v2_shots: List[Dict[str, Any]],
    a1_shots: List[Dict[str, Any]],
    output_path: str,
    track_states: Optional[Dict[str, Dict[str, bool]]] = None,
    crf: int = 18,
    preset: str = "fast",
    audio_bitrate: str = "192k",
    captions_path: Optional[str] = None,
) -> List[str]:
    """Build an FFmpeg filtergraph that matches the NLE timeline.

    Clip dictionaries are expected to carry:
      path, start_frame, duration_frames, trim_in, has_audio, gain, muted, hidden

    V1 starts are computed by the caller from `sequence_index`; this function
    treats them as absolute. V2/A1 use free placement.
    """
    track_states = track_states or {}

    def is_hidden(track_index: int) -> bool:
        state = track_states.get(str(track_index)) or track_states.get(track_index) or {}
        return bool(state.get("hidden"))

    def is_muted(track_index: int) -> bool:
        state = track_states.get(str(track_index)) or track_states.get(track_index) or {}
        return bool(state.get("muted"))

    all_shots = v1_shots + v2_shots + a1_shots
    max_frame = 0
    for shot in all_shots:
        end_frame = int(shot.get("start_frame", 0)) + max(0, int(shot.get("duration_frames", 0)))
        max_frame = max(max_frame, end_frame)
    duration_sec = max(1, max_frame) / max(1, project_fps)

    cmd = ["ffmpeg", "-y"]

    # Input 0: black canvas at the project duration.
    cmd.extend([
        "-f", "lavfi",
        "-i", f"color=c=black:s={project_width}x{project_height}:r={project_fps}:d={duration_sec}",
    ])

    input_idx = 1
    filter_chains: List[str] = []
    video_outputs: List[str] = []
    audio_outputs: List[str] = []

    def add_audio_chain(shot: Dict[str, Any], input_index: int, track_index: int) -> None:
        if not shot.get("has_audio") or shot.get("muted") or is_muted(track_index):
            return
        duration_frames = max(0, int(shot.get("duration_frames", 0)))
        if duration_frames <= 0:
            return
        trim_in_s = int(shot.get("trim_in", 0)) / project_fps
        duration_s = duration_frames / project_fps
        start_s = int(shot.get("start_frame", 0)) / project_fps
        delay_ms = max(0, int(round(start_s * 1000)))
        gain = float(shot.get("gain", 1.0) or 1.0)
        label = f"[a_{input_index}]"
        chain = f"[{input_index}:a]atrim=start={trim_in_s}:duration={duration_s},asetpts=PTS-STARTPTS"
        if abs(gain - 1.0) > 1e-3:
            chain += f",volume={gain}"
        chain += f",adelay={delay_ms}|{delay_ms}{label}"
        filter_chains.append(chain)
        audio_outputs.append(label)

    def process_video_track(shots: List[Dict[str, Any]], track_label: str, track_index: int, overlay: bool) -> List[str]:
        nonlocal input_idx
        outputs: List[str] = []
        if is_hidden(track_index):
            return outputs
        for idx, shot in enumerate(shots):
            if shot.get("hidden"):
                continue
            duration_frames = max(0, int(shot.get("duration_frames", 0)))
            if duration_frames <= 0:
                continue
            path = shot.get("path")
            if not path:
                continue
            start_s = int(shot.get("start_frame", 0)) / project_fps
            end_s = start_s + duration_frames / project_fps
            trim_in_s = int(shot.get("trim_in", 0)) / project_fps
            duration_s = duration_frames / project_fps

            cmd.extend(["-i", path])
            fc = f"[{input_idx}:v]trim=start={trim_in_s}:duration={duration_s},setpts=PTS-STARTPTS"
            fc += f",scale={project_width}:{project_height}:force_original_aspect_ratio=decrease"
            if overlay:
                # Transparent padding keeps V2 from blacking out V1.
                fc += (
                    f",pad={project_width}:{project_height}:(ow-iw)/2:(oh-ih)/2:color=black@0"
                    ",format=rgba"
                )
            else:
                fc += f",pad={project_width}:{project_height}:(ow-iw)/2:(oh-ih)/2:color=black"
            fc += f",setpts=PTS+{start_s}/TB"
            label = f"[{track_label}_{idx}]"
            fc += label
            filter_chains.append(fc)
            outputs.append((label, start_s, end_s))
            add_audio_chain(shot, input_idx, track_index)
            input_idx += 1
        return outputs

    v1_processed = process_video_track(v1_shots, "v1", 0, overlay=False)
    v2_processed = process_video_track(v2_shots, "v2", 1, overlay=True)

    # Standalone audio track(s). A1 is track 2.
    for idx, shot in enumerate(a1_shots):
        if is_hidden(2) or shot.get("hidden"):
            continue
        duration_frames = max(0, int(shot.get("duration_frames", 0)))
        if duration_frames <= 0 or not shot.get("path"):
            continue
        cmd.extend(["-i", shot["path"]])
        add_audio_chain(shot, input_idx, 2)
        input_idx += 1

    # Composite V1 over the black canvas.
    current_bg = "[0:v]"
    for idx, (vid_label, start_s, end_s) in enumerate(v1_processed):
        next_bg = f"[bg_v1_{idx}]"
        filter_chains.append(
            f"{current_bg}{vid_label}overlay=enable='between(t,{start_s},{end_s})':"
            f"eof_action=pass:format=auto{next_bg}"
        )
        current_bg = next_bg

    # Composite V2 over V1 with alpha-aware overlay.
    for idx, (vid_label, start_s, end_s) in enumerate(v2_processed):
        next_bg = f"[bg_v2_{idx}]"
        filter_chains.append(
            f"{current_bg}{vid_label}overlay=enable='between(t,{start_s},{end_s})':"
            f"eof_action=pass:format=auto{next_bg}"
        )
        current_bg = next_bg

    final_video_label = current_bg
    if captions_path:
        escaped = (
            str(captions_path)
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        filter_chains.append(f"{final_video_label}ass=filename='{escaped}'[v_captioned]")
        final_video_label = "[v_captioned]"

    final_audio_label = None
    if len(audio_outputs) == 1:
        final_audio_label = audio_outputs[0]
    elif len(audio_outputs) > 1:
        final_audio_label = "[a_out]"
        filter_chains.append(
            f"{''.join(audio_outputs)}amix=inputs={len(audio_outputs)}:"
            f"duration=longest:dropout_transition=0{final_audio_label}"
        )

    cmd.extend(["-filter_complex", ";".join(filter_chains)])
    cmd.extend(["-map", final_video_label])
    if final_audio_label:
        cmd.extend(["-map", final_audio_label])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", str(audio_bitrate),
        "-t", f"{duration_sec:.3f}",
        "-movflags", "+faststart",
        output_path,
    ])
    return cmd
