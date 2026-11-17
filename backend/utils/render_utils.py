import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _curve_expression(
    values: List[tuple],
    fps: float,
    default: float,
    variable: str,
) -> Optional[str]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return f"{ordered[0][1]:.4f}"

    fps = max(1.0, float(fps or 25))
    expr = f"{ordered[-1][1]:.4f}"
    for index in range(len(ordered) - 2, -1, -1):
        f0, o0 = ordered[index]
        f1, o1 = ordered[index + 1]
        t0, t1 = f0 / fps, f1 / fps
        if t1 <= t0:
            continue
        slope = (o1 - o0) / (t1 - t0)
        segment = f"({o0:.4f}+({slope:.6f})*({variable}-{t0:.6f}))"
        expr = f"if(lt({variable},{t1:.6f}),{segment},{expr})"

    f0, o0 = ordered[0]
    t0 = f0 / fps
    return f"if(lt({variable},{t0:.6f}),{o0:.4f},{expr})"


def _keyframe_points(
    keyframes: List[Dict[str, Any]],
    key: str,
    default: float,
    clamp_min: float,
    clamp_max: float,
) -> List[tuple]:
    points: Dict[int, float] = {}
    for raw in keyframes or []:
        try:
            frame = max(0, int(raw.get("frame") or 0))
            value = max(clamp_min, min(clamp_max, float(raw.get(key, default))))
        except (TypeError, ValueError):
            continue
        points[frame] = value
    return sorted(points.items())


def opacity_expression(keyframes: List[Dict[str, Any]], fps: float) -> Optional[str]:
    """Piecewise-linear opacity curve in clip-local seconds.

    Frames are local to the visible clip (0 = first visible frame), which is why
    this is evaluated before the clip's PTS is shifted onto the timeline.
    """
    points = _keyframe_points(keyframes, "opacity", 1.0, 0.0, 1.0)
    if all(abs(value - 1.0) < 1e-6 for _, value in points):
        return None
    return _curve_expression(points, fps, 1.0, "T")


def volume_expression(keyframes: List[Dict[str, Any]], fps: float) -> Optional[str]:
    points = _keyframe_points(keyframes, "volume", 1.0, 0.0, 2.0)
    if all(abs(value - 1.0) < 1e-6 for _, value in points):
        return None
    return _curve_expression(points, fps, 1.0, "t")


def transform_expressions(
    keyframes: List[Dict[str, Any]],
    fps: float,
) -> Optional[Dict[str, str]]:
    scales = _keyframe_points(keyframes, "scale", 1.0, 0.1, 4.0)
    xs = _keyframe_points(keyframes, "x", 0.0, -1.0, 1.0)
    ys = _keyframe_points(keyframes, "y", 0.0, -1.0, 1.0)
    if not scales and not xs and not ys:
        return None
    if (
        all(abs(value - 1.0) < 1e-6 for _, value in scales)
        and all(abs(value) < 1e-6 for _, value in xs)
        and all(abs(value) < 1e-6 for _, value in ys)
    ):
        return None
    return {
        "scale": _curve_expression(scales, fps, 1.0, "time") or "1.0000",
        "x": _curve_expression(xs, fps, 0.0, "time") or "0.0000",
        "y": _curve_expression(ys, fps, 0.0, "time") or "0.0000",
    }


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
    audio_normalize: bool = False,
    captions_path: Optional[str] = None,
    start_seconds: float = 0.0,
    duration_seconds: Optional[float] = None,
    fit: str = "pad",
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

    def has_solo() -> bool:
        return any(
            bool((state or {}).get("solo"))
            for state in track_states.values()
        )

    def is_muted(track_index: int) -> bool:
        state = track_states.get(str(track_index)) or track_states.get(track_index) or {}
        if state.get("solo"):
            return False
        if has_solo():
            return True
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
    audio_outputs: List[str] = []

    def add_audio_chain(shot: Dict[str, Any], input_index: int, track_index: int) -> None:
        if not shot.get("has_audio") or shot.get("muted") or is_muted(track_index):
            return
        duration_frames = max(0, int(shot.get("duration_frames", 0)))
        if duration_frames <= 0:
            return
        source_frames = max(1, int(shot.get("source_frames", duration_frames)))
        speed = min(4.0, max(0.25, float(shot.get("speed", 1.0) or 1.0)))
        trim_in_s = int(shot.get("trim_in", 0)) / project_fps
        source_duration_s = source_frames / project_fps
        start_s = int(shot.get("start_frame", 0)) / project_fps
        delay_ms = max(0, int(round(start_s * 1000)))
        gain = float(shot.get("gain", 1.0) or 1.0)
        label = f"[a_{input_index}]"
        chain = (
            f"[{input_index}:a]atrim=start={trim_in_s}:duration={source_duration_s}"
            ",asetpts=PTS-STARTPTS"
        )
        if abs(speed - 1.0) > 1e-3:
            chain += f",atempo={speed}"
        if abs(gain - 1.0) > 1e-3:
            chain += f",volume={gain}"
        output_duration = duration_frames / project_fps
        fade_in = min(output_duration, max(0.0, float(shot.get("fade_in", 0) or 0)))
        fade_out = min(output_duration, max(0.0, float(shot.get("fade_out", 0) or 0)))
        volume_expr = volume_expression(shot.get("keyframes") or [], project_fps)
        if volume_expr:
            chain += f",volume=eval=frame:volume='{volume_expr}'"
        if fade_in > 0:
            chain += f",afade=t=in:st=0:d={fade_in:.3f}"
        if fade_out > 0:
            chain += (
                f",afade=t=out:st={max(0.0, output_duration - fade_out):.3f}"
                f":d={fade_out:.3f}"
            )
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
            source_frames = max(1, int(shot.get("source_frames", duration_frames)))
            speed = min(4.0, max(0.25, float(shot.get("speed", 1.0) or 1.0)))
            start_s = int(shot.get("start_frame", 0)) / project_fps
            end_s = start_s + duration_frames / project_fps
            trim_in_s = int(shot.get("trim_in", 0)) / project_fps
            source_duration_s = source_frames / project_fps

            cmd.extend(["-i", path])
            fc = (
                f"[{input_idx}:v]trim=start={trim_in_s}:duration={source_duration_s}"
                f",setpts=(PTS-STARTPTS)/{speed}"
            )
            if fit == "crop":
                fc += (
                    f",scale={project_width}:{project_height}:force_original_aspect_ratio=increase"
                    f",crop={project_width}:{project_height}"
                )
                if overlay:
                    fc += ",format=rgba"
            else:
                fc += f",scale={project_width}:{project_height}:force_original_aspect_ratio=decrease"
                if overlay:
                    # Transparent padding keeps V2 from blacking out V1.
                    fc += (
                        f",pad={project_width}:{project_height}:(ow-iw)/2:(oh-ih)/2:color=black@0"
                        ",format=rgba"
                    )
                else:
                    fc += f",pad={project_width}:{project_height}:(ow-iw)/2:(oh-ih)/2:color=black"

            keyframes = shot.get("keyframes") or []
            transform = transform_expressions(keyframes, project_fps)
            if transform:
                fc += (
                    f",zoompan=z='clip({transform['scale']},0.1,4)'"
                    f":x='max(0,min(iw-iw/zoom,(iw-iw/zoom)/2+({transform['x']})*iw/2))'"
                    f":y='max(0,min(ih-ih/zoom,(ih-ih/zoom)/2+({transform['y']})*ih/2))'"
                    f":d=1:s={project_width}x{project_height}:fps={project_fps}"
                    ",format=rgba"
                )

            opacity = opacity_expression(keyframes, project_fps)
            if opacity:
                if not overlay and not transform:
                    fc += ",format=rgba"
                fc += (
                    ",geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                    f"a='alpha(X,Y)*clip(({opacity}),0,1)'"
                )

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

    if final_audio_label and audio_normalize:
        filter_chains.append(
            f"{final_audio_label}loudnorm=I=-16:TP=-1.5:LRA=11[a_norm]"
        )
        final_audio_label = "[a_norm]"

    cmd.extend(["-filter_complex", ";".join(filter_chains)])
    cmd.extend(["-map", final_video_label])
    if final_audio_label:
        cmd.extend(["-map", final_audio_label])

    output_duration = duration_seconds if duration_seconds is not None else duration_sec
    if start_seconds > 0:
        cmd.extend(["-ss", f"{max(0.0, start_seconds):.3f}"])
    cmd.extend([
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", str(audio_bitrate),
        "-t", f"{max(0.05, output_duration):.3f}",
        "-movflags", "+faststart",
        output_path,
    ])
    return cmd
