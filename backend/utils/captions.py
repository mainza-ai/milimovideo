"""Generate ASS subtitle files from project caption cues.

Local-first by construction: no network, no external service. libass is already
part of the FFmpeg build used for the render path (verified with `-filters`).
"""

import os
import tempfile
from typing import Any, Dict, List, Optional


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole = int(seconds % 60)
    hundredths = int(round((seconds - int(seconds)) * 100))
    if hundredths >= 100:
        whole += 1
        hundredths = 0
    return f"{hours}:{minutes:02d}:{whole:02d}.{hundredths:02d}"


def _escape(text: str) -> str:
    # ASS treats { } as override blocks and backslash as an escape introducer.
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\n")
        .replace("\n", "\\N")
        .strip()
    )


def build_ass(
    captions: List[Dict[str, Any]],
    width: int,
    height: int,
    fps: float,
) -> str:
    font_size = max(16, int(round(width * 0.042)))
    margin_v = max(18, int(round(height * 0.075)))
    outline = max(2, int(round(font_size * 0.09)))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,{outline},0,"
            f"2,60,60,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    fps = max(1.0, float(fps or 25))
    for cue in captions or []:
        text = _escape(cue.get("text") or "")
        if not text:
            continue
        start = max(0, int(cue.get("start_frame") or 0)) / fps
        end = max(start + 0.05, int(cue.get("end_frame") or 0) / fps)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
        )

    return "\n".join(lines) + "\n"


def write_ass_file(
    captions: List[Dict[str, Any]],
    width: int,
    height: int,
    fps: float,
) -> Optional[str]:
    if not captions:
        return None
    with tempfile.NamedTemporaryFile(
        "w", suffix=".ass", prefix="milimo-captions-", delete=False, encoding="utf-8",
    ) as handle:
        handle.write(build_ass(captions, width, height, fps))
        return handle.name


def cleanup(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
