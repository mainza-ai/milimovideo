#!/usr/bin/env python
"""Repeatable performance/memory baseline for this workstation.

This does not load a model. It records the memory plane, cost estimates,
preflight decisions, and a real FFmpeg software-vs-VideoToolbox encode so a
regression has a before/after artifact.

    ./milimov/bin/python backend/benchmark_performance.py
    ./milimov/bin/python backend/benchmark_performance.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ.setdefault("MILIMO_MEMORY_ENFORCE", "0")

from services.memory import budget, costs  # noqa: E402

OUTPUT = ROOT / "benchmarks" / "performance.json"


def memory_rows() -> dict:
    snapshot = budget.snapshot({}, [])
    return {
        "device": snapshot.device,
        "total_gb": snapshot.total_gb,
        "device_recommended_gb": snapshot.device_recommended_gb,
        "process_cap_gb": snapshot.process_cap_gb,
        "physical_available_gb": snapshot.physical_available_gb,
        "swap_free_gb": snapshot.swap_free_gb,
        "swap_percent": snapshot.swap_percent,
        "pressure": snapshot.pressure,
        "allocator_current_gb": snapshot.allocator_current_gb,
        "allocator_cache_gb": snapshot.allocator_cache_gb,
        "notes": snapshot.notes,
        "estimates": {
            "video": costs.estimate("video", {}).as_dict(),
            "image": costs.estimate("image", {}).as_dict(),
        },
    }


def encode_row(ffmpeg: str) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="milimo-perf-") as tmp:
        source = Path(tmp) / "source.mp4"
        subprocess.run(
            [
                ffmpeg, "-y", "-f", "lavfi", "-i",
                "testsrc2=size=1920x1080:rate=30:duration=5",
                "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
                str(source),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        candidates = [
            ("libx264", ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]),
            ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-q:v", "65"]),
        ]
        for label, args in candidates:
            output = Path(tmp) / f"{label}.mp4"
            started = time.perf_counter()
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", str(source), *args, "-pix_fmt", "yuv420p", str(output)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                rows.append({"encoder": label, "ok": False})
                continue
            elapsed = time.perf_counter() - started
            rows.append({
                "encoder": label,
                "ok": True,
                "wall_s": round(elapsed, 3),
                "bytes": output.stat().st_size,
            })
    return rows


def api_rows(api: str) -> dict | None:
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(base_url=api.rstrip("/"), timeout=5.0) as client:
            started = time.perf_counter()
            health = client.get("/health")
            health_ms = round((time.perf_counter() - started) * 1000, 2)
            started = time.perf_counter()
            projects = client.get("/projects")
            projects_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "health_status": health.status_code,
            "health_ms": health_ms,
            "projects_status": projects.status_code,
            "projects_ms": projects_ms,
            "projects": len(projects.json()) if projects.ok else None,
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


def markdown(report: dict) -> str:
    mem = report["memory"]
    lines = [
        f"# Performance baseline — {report['captured_at']}",
        "",
        f"- Host: {report['host']['machine']} · {report['host']['platform']}",
        f"- Python/torch: {report['host']['python']} / {report['host']['torch']}",
        f"- Device: {mem['device']} · recommended {mem['device_recommended_gb']:.1f} GB · process cap {mem['process_cap_gb']:.1f} GB",
        f"- Physical free: {mem['physical_available_gb']:.1f} GB · swap free {mem['swap_free_gb']:.1f} GB ({mem['swap_percent']:.0f}%) · pressure {mem['pressure']}",
        f"- Allocator: current {mem['allocator_current_gb']:.2f} GB · cache {mem['allocator_cache_gb']:.2f} GB",
        "",
        "| Encode | Wall s | Bytes |",
        "|---|---:|---:|",
    ]
    for row in report["encode"]:
        if row.get("ok"):
            lines.append(f"| {row['encoder']} | {row['wall_s']} | {row['bytes']:,} |")
        else:
            lines.append(f"| {row['encoder']} | failed | — |")
    if report.get("api"):
        lines += ["", f"API: `{json.dumps(report['api'])}`"]
    lines += ["", "## Estimates", ""]
    for lane, estimate in mem["estimates"].items():
        lines.append(f"- {lane}: {estimate['total_gb']} GB ({estimate['base_gb']} base + {estimate['workspace_gb']} workspace)")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="Optional running backend URL for API latency")
    args = parser.parse_args()
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "unavailable"
    ffmpeg = shutil.which("ffmpeg")
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch_version,
            "ffmpeg": ffmpeg,
        },
        "memory": memory_rows(),
        "encode": encode_row(ffmpeg) if ffmpeg else [],
        "api": api_rows(args.api) if args.api else None,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    summary = OUTPUT.with_suffix(".md")
    summary.write_text(markdown(report))
    print(markdown(report))
    print(f"Wrote {OUTPUT} and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
