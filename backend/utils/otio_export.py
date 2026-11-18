"""OpenTimelineIO export for the computed cut.

The finishing NLE is the right place for color, titles, and the final mix. This
builds an OTIO timeline from the same shot fields the renderer uses: magnetic V1,
free V2/A1, trims, and speed. Missing media is preserved as a MissingReference so
the edit survives an offline drive.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import opentimelineio as otio

from file_utils import resolve_path


def _rate(fps: int) -> float:
    return float(fps or 25)


def _external_reference(path: Optional[str], source_frames: int, fps: float):
    if path and os.path.exists(path):
        return otio.schema.ExternalReference(
            target_url=Path(path).resolve().as_uri(),
            available_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, fps),
                otio.opentime.RationalTime(max(0, source_frames), fps),
            ),
        )
    return otio.schema.MissingReference(
        available_range=otio.opentime.TimeRange(
            otio.opentime.RationalTime(0, fps),
            otio.opentime.RationalTime(max(0, source_frames), fps),
        ),
    )


def _clip_for_shot(shot, fps: float, start_frame: int):
    trim_in = max(0, int(shot.trim_in or 0))
    trim_out = max(0, int(shot.trim_out or 0))
    total = max(0, int(shot.num_frames or 121))
    source_frames = max(1, total - trim_in - trim_out)
    speed = min(4.0, max(0.25, float(shot.speed or 1.0)))
    duration = max(1, round(source_frames / speed))

    resolved = resolve_path(shot.video_url) if shot.video_url else None
    media = _external_reference(resolved, total, fps)
    clip = otio.schema.Clip(
        name=(shot.action or shot.prompt or shot.id)[:80],
        media_reference=media,
        source_range=otio.opentime.TimeRange(
            otio.opentime.RationalTime(trim_in, fps),
            otio.opentime.RationalTime(source_frames, fps),
        ),
    )
    if abs(speed - 1.0) > 0.001:
        clip.effects.append(otio.schema.LinearTimeWarp(name="speed", time_scalar=speed))
    clip.metadata["milimo"] = {
        "shot_id": shot.id,
        "prompt": shot.prompt,
        "seed": shot.seed,
        "num_frames": total,
        "trim_in": trim_in,
        "trim_out": trim_out,
        "speed": speed,
        "last_job_id": shot.last_job_id,
        "track_index": shot.track_index,
        "transition_in": int(getattr(shot, "transition_in", 0) or 0),
    }
    return clip, start_frame, duration


def build_otio_timeline(project, shots: Iterable, fps: Optional[int] = None):
    """Build a 4-track OTIO timeline: V1 (magnetic), V2, A1, A2.

    V1 start frames are computed in sequence order; V2/A1/A2 use ``start_frame``.
    Audio-bearing video clips live on V1/V2; ``track_index`` 2/3 are A1/A2. The NLE is expected to relink media if the checkout moved.

    Speed is represented as a ``LinearTimeWarp`` effect plus ``metadata.milimo``.
    OTIO readers that do not apply effects to track layout may show the source
    duration for those clips; the NLE applies the effect on import. Keep speed
    clips at full speed if a strict OTIO round-trip is required.
    """
    rate = _rate(fps or project.fps)
    ordered = sorted(
        [shot for shot in shots if shot.video_url],
        key=lambda shot: (
            shot.sequence_index is None,
            shot.sequence_index if shot.sequence_index is not None else 0,
            shot.created_at,
            shot.id,
        ),
    )

    v1 = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v2 = otio.schema.Track(name="V2", kind=otio.schema.TrackKind.Video)
    a1 = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    a2 = otio.schema.Track(name="A2", kind=otio.schema.TrackKind.Audio)

    v1_cursor = 0
    v1_previous_duration = 0
    v2_cursor = 0
    a1_cursor = 0
    a2_cursor = 0
    end_frame = 0

    def append_at(track, clip, start, cursor, rate):
        gap = max(0, start - cursor)
        if gap:
            track.append(otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    otio.opentime.RationalTime(0, rate),
                    otio.opentime.RationalTime(gap, rate),
                ),
            ))
        track.append(clip)
        return max(start, cursor) + int(clip.source_range.duration.value)

    for shot in ordered:
        if shot.track_index == 0:
            transition = 0
            if v1_previous_duration:
                source_frames = max(
                    1,
                    max(0, int(shot.num_frames or 121))
                    - max(0, int(shot.trim_in or 0))
                    - max(0, int(shot.trim_out or 0)),
                )
                speed = min(4.0, max(0.25, float(shot.speed or 1.0)))
                duration = max(1, round(source_frames / speed))
                transition = min(
                    max(0, int(getattr(shot, "transition_in", 0) or 0)),
                    duration // 2,
                    v1_previous_duration // 2,
                )
            v1_cursor = max(0, v1_cursor - transition)
            clip, start, duration = _clip_for_shot(shot, rate, v1_cursor)
            v1.append(clip)
            v1_cursor = start + duration
            v1_previous_duration = duration
            end_frame = max(end_frame, v1_cursor)
        elif shot.track_index == 1:
            start = max(0, int(shot.start_frame or 0))
            clip, start, duration = _clip_for_shot(shot, rate, start)
            v2_cursor = append_at(v2, clip, start, v2_cursor, rate)
            end_frame = max(end_frame, v2_cursor)
        elif shot.track_index == 2:
            start = max(0, int(shot.start_frame or 0))
            clip, start, duration = _clip_for_shot(shot, rate, start)
            a1_cursor = append_at(a1, clip, start, a1_cursor, rate)
            end_frame = max(end_frame, a1_cursor)
        elif shot.track_index == 3:
            start = max(0, int(shot.start_frame or 0))
            clip, start, duration = _clip_for_shot(shot, rate, start)
            a2_cursor = append_at(a2, clip, start, a2_cursor, rate)
            end_frame = max(end_frame, a2_cursor)

    timeline = otio.schema.Timeline(name=project.name or "Milimo cut")
    stack = otio.schema.Stack(name="Milimo cut")
    stack.append(v1)
    stack.append(v2)
    stack.append(a1)
    stack.append(a2)
    timeline.tracks = stack
    timeline.global_start_time = otio.opentime.RationalTime(0, rate)
    timeline.metadata["milimo"] = {
        "project_id": project.id,
        "project_name": project.name,
        "revision": getattr(project, "revision", None),
        "fps": rate,
        "duration_frames": end_frame,
        "source": "milimo-video",
    }
    return timeline


def otio_json(project, shots: Iterable, fps: Optional[int] = None) -> str:
    return otio.adapters.write_to_string(build_otio_timeline(project, shots, fps), "otio_json")
