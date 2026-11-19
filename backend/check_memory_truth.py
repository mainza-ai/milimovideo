#!/usr/bin/env python
"""CI gate: reported memory ceiling must be the real admission cap.

Fails when the snapshot claims more headroom than the process or the machine can
provide, or when admission would accept a job larger than the cap. This is the
M0 acceptance check in executable form; it does not load a model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ.setdefault("MILIMO_MEMORY_ENFORCE", "1")

from services.memory import budget  # noqa: E402


def main() -> int:
    snapshot = budget.snapshot({}, [])
    problems: list[str] = []

    if snapshot.ceiling_gb > snapshot.total_gb + 0.01:
        problems.append(
            f"ceiling {snapshot.ceiling_gb:.2f} GB exceeds machine total {snapshot.total_gb:.2f} GB"
        )
    if snapshot.device == "mps":
        if snapshot.process_cap_gb <= 0:
            problems.append("MPS process cap is missing")
        if snapshot.ceiling_gb > snapshot.process_cap_gb + 0.01:
            problems.append(
                f"ceiling {snapshot.ceiling_gb:.2f} GB exceeds process cap {snapshot.process_cap_gb:.2f} GB"
            )
        if snapshot.process_cap_gb > snapshot.device_recommended_gb + 0.01:
            problems.append(
                f"process cap {snapshot.process_cap_gb:.2f} GB exceeds recommended {snapshot.device_recommended_gb:.2f} GB"
            )

    # A deliberately oversized job must be refused with a retryable reason, not
    # admitted and then OOM-killed.
    refusal = budget.admit(
        "video",
        {"num_frames": 121, "width": 1920, "height": 1080, "upscale": True},
    )
    if refusal.ok and refusal.estimate and refusal.estimate.total_gb > snapshot.ceiling_gb:
        problems.append(
            f"admission accepted {refusal.estimate.total_gb:.2f} GB above the {snapshot.ceiling_gb:.2f} GB cap"
        )

    print(json.dumps({
        "device": snapshot.device,
        "total_gb": round(snapshot.total_gb, 2),
        "device_recommended_gb": round(snapshot.device_recommended_gb, 2),
        "process_cap_gb": round(snapshot.process_cap_gb, 2),
        "ceiling_gb": round(snapshot.ceiling_gb, 2),
        "physical_available_gb": round(snapshot.physical_available_gb, 2),
        "swap_free_gb": round(snapshot.swap_free_gb, 2),
        "oversized_refused": not refusal.ok,
        "problems": problems,
    }, indent=2))

    if problems:
        for problem in problems:
            print(f"::error::{problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
