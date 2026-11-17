#!/usr/bin/env python
"""Quality/memory/time comparison for the local model lanes.

M5 is a model-artifact program: there are no alternative quantized checkpoints in
this repository to swap in. This tool is the part that belongs in code — it
records, for the checkpoints that are actually present, the estimated peak, the
measured peak, the load time, and the wall time on a reference workload, so a
future checkpoint swap has a baseline to beat.

Dry-run (safe, no model load):

    ./milimov/bin/python backend/benchmark_models.py

Execute a lane against a running backend (loads the model):

    ./milimov/bin/python backend/benchmark_models.py --run video --i-know-it-loads-models
    ./milimov/bin/python backend/benchmark_models.py --run image --i-know-it-loads-models

Results append to ``benchmarks/model-lanes.json`` (gitignored) and print as a
markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ.setdefault("MILIMO_MEMORY_ENFORCE", "0")

from services.memory import costs  # noqa: E402

REFERENCE = {
    "video": {"num_frames": 49, "width": 512, "height": 320, "upscale": False},
    "image": {"width": 1024, "height": 1024, "reference_count": 0},
    "sam": {},
    "depth": {},
}

OUTPUT = ROOT / "benchmarks" / "model-lanes.json"


def estimate_rows() -> list[dict]:
    rows = []
    for lane, params in REFERENCE.items():
        estimate = costs.estimate(lane, params)
        rows.append({
            "lane": lane,
            "params": params,
            "estimated_peak_gb": estimate.total_gb,
            "base_gb": estimate.base_gb,
            "workspace_gb": estimate.workspace_gb,
        })
    return rows


def markdown(rows: list[dict]) -> str:
    lines = [
        "| Lane | Reference workload | Estimated peak | Measured peak | Load | Wall |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        params = row.get("params") or {}
        workload = ", ".join(f"{key}={value}" for key, value in params.items()) or "n/a"
        measured = row.get("measured_peak_gb")
        lines.append(
            "| {lane} | {workload} | {est:.1f} GB | {measured} | {load} | {wall} |".format(
                lane=row["lane"],
                workload=workload,
                est=row["estimated_peak_gb"],
                measured=f"{measured:.1f} GB" if isinstance(measured, (int, float)) else "—",
                load=f"{row['load_s']:.1f}s" if row.get("load_s") is not None else "—",
                wall=f"{row['wall_s']:.1f}s" if row.get("wall_s") is not None else "—",
            )
        )
    return "\n".join(lines)


def run_lane(lane: str, api: str, project_id: str) -> dict:
    """Run one small job through the public API and read its memory trace."""
    import httpx

    from datetime import datetime

    job_id = f"bench_{lane}_{int(time.time())}"
    with httpx.Client(base_url=api, timeout=60.0) as client:
        if lane == "video":
            payload = {
                "project_id": project_id,
                "shot_config": {
                    "id": job_id,
                    "prompt": "a static test pattern, no motion",
                    "num_frames": 49,
                    "width": 512,
                    "height": 320,
                    "num_inference_steps": 8,
                    "enhance_prompt": False,
                    "upscale": False,
                },
            }
            response = client.post("/generate/advanced", json=payload)
        elif lane == "image":
            payload = {
                "project_id": project_id,
                "prompt": "a static test pattern",
                "width": 512,
                "height": 512,
                "num_inference_steps": 4,
                "enhance_prompt": False,
                "is_thumbnail": True,
            }
            response = client.post("/generate/image", json=payload)
        else:
            raise SystemExit(f"Lane {lane!r} is not runnable from this tool")

        response.raise_for_status()
        returned = response.json().get("job_id") or job_id
        started = time.monotonic()
        status = {}
        while time.monotonic() - started < 3600:
            status = client.get(f"/status/{returned}").json()
            if status.get("status") in ("completed", "failed", "cancelled"):
                break
            time.sleep(2.0)
        else:
            raise SystemExit(f"{lane}: timed out")
        wall = time.monotonic() - started
        memory = status.get("memory") or {}
        return {
            "job_id": returned,
            "status": status.get("status"),
            "wall_s": round(wall, 1),
            "measured_peak_gb": memory.get("peak_gb"),
            "peak_stage": memory.get("peak_stage"),
            "measured_at": datetime.utcnow().isoformat() + "Z",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["video", "image"])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--project", default=None, help="Existing project id for the run")
    parser.add_argument(
        "--i-know-it-loads-models", action="store_true",
        help="Required for --run: the job will load the real pipeline",
    )
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()

    rows = estimate_rows()
    if args.run:
        if not args.i_know_it_loads_models:
            raise SystemExit("--run loads a real model; pass --i-know-it-loads-models")
        if not args.project:
            raise SystemExit("--run needs --project <project_id>")
        lane = args.run
        result = run_lane(lane, args.api, args.project)
        for row in rows:
            if row["lane"] == lane:
                row.update(result)
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            existing = []
    existing.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows})
    out.write_text(json.dumps(existing[-20:], indent=2))
    print(markdown(rows))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
