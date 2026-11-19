"""Child-process memory truth (ffmpeg, PyAV, Comfy, SAM helpers).

The unified budget accounts for models loaded in this process and for declared
external engines, but an encoder child is invisible to both. A 4K x264 encode can
hold a gigabyte of buffers while the job's model slot still reads as the only
consumer. This module measures the process tree so Setup and `/system/memory` can
show the real total and the pressure guard can include it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import psutil

_CACHE: dict[str, Any] = {"at": 0.0, "rows": [], "summary": {}}
_TTL_SEC = 2.0
_LOCK = threading.Lock()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["rss_gb"] for row in rows)
    ffmpeg = [row for row in rows if row["name"].startswith("ffmpeg")]
    return {
        "count": len(rows),
        "total_gb": round(total, 3),
        "ffmpeg_count": len(ffmpeg),
        "ffmpeg_gb": round(sum(row["rss_gb"] for row in ffmpeg), 3),
        "threads": sum(row["threads"] for row in rows),
    }


def snapshot(force: bool = False) -> dict[str, Any]:
    """Measured descendants of this backend, lightly cached."""
    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE["rows"] and now - _CACHE["at"] < _TTL_SEC:
            return {"processes": _CACHE["rows"], "summary": _CACHE["summary"]}

    rows: list[dict[str, Any]] = []
    try:
        parent = psutil.Process()
        for child in parent.children(recursive=True):
            try:
                info = child.memory_info()
                rows.append(
                    {
                        "pid": child.pid,
                        "name": child.name(),
                        "rss_gb": round(float(info.rss) / 1e9, 3),
                        "threads": child.num_threads(),
                        "status": child.status(),
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        rows = []

    summary = _summarize(rows)
    with _LOCK:
        _CACHE.update({"at": now, "rows": rows, "summary": summary})
    return {"processes": rows[:12], "summary": summary}


def ffmpeg_threads() -> int:
    """Thread count already spent on encoders, for x264 thread budgeting."""
    summary = snapshot()["summary"]
    if not summary.get("ffmpeg_count"):
        return 0
    return int(summary.get("threads") or 0)
