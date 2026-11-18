"""Import an OpenTimelineIO cut as Milimo shots.

Round-trips the export: V1 becomes magnetic sequence order, V2/A1 keep their
start frames, trims/speed are preserved, and project-local media URLs are
rewritten back to ``/projects/...``. Importing does not mutate the database; the
caller applies the returned shot list as one document edit.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import opentimelineio as otio

import config


def _media_url(target_url: Optional[str]) -> Optional[str]:
    if not target_url:
        return None
    if target_url.startswith("file://"):
        path = target_url[len("file://"):]
        projects = os.path.realpath(config.PROJECTS_DIR)
        absolute = os.path.realpath(path)
        if absolute.startswith(projects + os.sep):
            rel = os.path.relpath(absolute, projects).replace(os.sep, "/")
            return f"/projects/{rel}"
        return absolute
    return target_url


def _speed(clip) -> float:
    for effect in clip.effects:
        if effect.effect_name == "LinearTimeWarp":
            try:
                return max(0.25, min(4.0, float(effect.time_scalar)))
            except (TypeError, ValueError):
                pass
    return 1.0


def _clip_shot(clip, project_id: str, track_index: int, index: int, start_frame: int) -> dict[str, Any]:
    source = clip.source_range
    available = clip.media_reference.available_range if clip.media_reference else None
    total = int(available.duration.value) if available else int(source.duration.value)
    trim_in = int(source.start_time.value)
    source_frames = int(source.duration.value)
    trim_out = max(0, total - trim_in - source_frames)
    meta = (clip.metadata or {}).get("milimo", {})
    media = _media_url(getattr(clip.media_reference, "target_url", None))
    return {
        "id": f"shot-{uuid.uuid4().hex[:12]}",
        "project_id": project_id,
        "index": index,
        "sequence_index": index,
        "prompt": meta.get("prompt") or clip.name or "",
        "negative_prompt": "",
        "seed": int(meta.get("seed") or 42),
        "width": int(meta.get("width") or 768),
        "height": int(meta.get("height") or 512),
        "num_frames": max(1, total),
        "fps": None,
        "cfg_scale": 2.0,
        "enhance_prompt": False,
        "upscale": False,
        "pipeline_override": "auto",
        "track_index": track_index,
        "start_frame": max(0, start_frame),
        "trim_in": trim_in,
        "trim_out": trim_out,
        "speed": _speed(clip),
        "timeline": [],
        "status": "pending",
        "video_url": media,
        "last_job_id": meta.get("last_job_id"),
    }


def shots_from_otio(project_id: str, payload: str | bytes, format: str | None = None) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if format is None:
        head = payload.lstrip()[:200].lower()
        format = "fcpx_xml" if head.startswith(("<?xml", "<fcpxml")) or "<fcpxml" in head else "otio_json"
    timeline = otio.adapters.read_from_string(payload, format)
    rate = timeline.global_start_time.rate if timeline.global_start_time else 25
    fps = int(round(rate)) or 25
    shots: list[dict[str, Any]] = []
    counter = 0

    for track in timeline.tracks:
        name = (track.name or "").upper()
        if track.kind == otio.schema.TrackKind.Audio or name.startswith("A"):
            track_index = 2
        elif name.startswith("V2") or name.startswith("V 2"):
            track_index = 1
        else:
            track_index = 0

        cursor = 0
        for item in track:
            if isinstance(item, otio.schema.Gap):
                cursor += int(item.duration().value)
                continue
            if not isinstance(item, otio.schema.Clip):
                continue
            source_frames = int(item.source_range.duration.value)
            speed = _speed(item)
            timeline_frames = max(1, round(source_frames / speed))
            start = 0 if track_index == 0 else cursor
            shots.append(_clip_shot(item, project_id, track_index, counter, start))
            counter += 1
            cursor = start + timeline_frames

    shots.sort(key=lambda shot: (shot["track_index"], shot["sequence_index"]))
    return {
        "name": timeline.name or "Imported cut",
        "fps": fps,
        "shots": shots,
        "duration_frames": max((shot["start_frame"] + shot["num_frames"] for shot in shots), default=0),
    }
