#!/usr/bin/env python
"""Small, honest evaluation of project memory retrieval.

The suite mirrors real studio questions: which take was approved, why was a shot
rejected, what does the character wear, what changed after a note was edited.
It measures top-1/top-3 hit rate, tokens packed, and latency on a throwaway
in-memory database, so CI can gate retrieval quality without a model or network.

    ./milimov/bin/python backend/agent_memory_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ.setdefault("MILIMO_MEMORY_ENFORCE", "0")

from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

import agent_memory  # noqa: E402
from database import AgentMemory  # noqa: E402

TASKS = [
    # (query, expected substring, shot_id)
    ("which take did the director approve for the rooftop jump", "Take 4 approved", None),
    ("why was the alley shot rejected", "Too much camera shake", None),
    ("what does the courier wear", "red raincoat", None),
    ("where does the chase end", "wet rooftop at dawn", None),
    ("which lens do we use for close-ups", "85mm", None),
    ("who is the antagonist", "Mara", None),
    ("what is the color grade for act two", "amber and teal", None),
    ("how long is the rooftop shot", "121 frames", None),
    ("what sound does the door make", "hydraulic hiss", None),
    ("which take is the favorite", "Take 4 approved", None),
    ("what did the reviewer dislike", "Too much camera shake", None),
    ("what is the rule for crowd shots", "no more than twelve extras", None),
    ("which shot uses the rain machine", "rain machine", "shot-rain"),
    ("what replaced the blue coat", "red raincoat", None),
    ("what is the act three location", "drainage tunnel", None),
    ("which camera move for the reveal", "slow push in", None),
    ("what is the title card font", "Söhne", None),
    ("how many frames per chunk", "chunk of 41 frames", None),
    ("what is the voice of the narrator", "low and dry", None),
    ("which project prefers 16:9", "16:9", None),
]

FACTS = [
    ("Take 4 approved for the rooftop jump; the fall reads clean.", "episodic", None),
    ("The alley shot was rejected: Too much camera shake in the turn.", "episodic", None),
    ("The courier wears a red raincoat in every act.", "semantic", None),
    ("The chase ends on the wet rooftop at dawn.", "semantic", None),
    ("Close-ups use an 85mm lens with a shallow stop.", "procedural", None),
    ("The antagonist is Mara, a fixer with a quiet voice.", "semantic", None),
    ("Act two is graded amber and teal; act one stays neutral.", "semantic", None),
    ("The rooftop shot is 121 frames at 25 fps.", "semantic", None),
    ("The door makes a hydraulic hiss, never a creak.", "semantic", None),
    ("Crowd scenes use no more than twelve extras.", "procedural", None),
    ("The rain machine sits frame-left on the rooftop.", "episodic", "shot-rain"),
    ("Act three moves into the drainage tunnel.", "semantic", None),
    ("The reveal uses a slow push in, no whip pan.", "procedural", None),
    ("Title cards use the Söhne typeface, 64 px.", "procedural", None),
    ("Generate in a chunk of 41 frames on 32 GB machines.", "procedural", None),
    ("The narrator is low and dry, close to the mic.", "semantic", None),
    ("Delivery is 16:9 at 1920x1080 for this project.", "procedural", None),
]


def build_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE VIRTUAL TABLE agentmemory_fts "
            "USING fts5(text, content='agentmemory', content_rowid='rowid')"
        )
        connection.exec_driver_sql(
            "CREATE TRIGGER agentmemory_ai AFTER INSERT ON agentmemory BEGIN "
            "INSERT INTO agentmemory_fts(rowid, text) VALUES (new.rowid, new.text); END"
        )
    return Session(engine)


def run_suite() -> dict:
    session = build_session()
    project_id = "eval-project"
    for text, kind, shot_id in FACTS:
        agent_memory.remember(
            session,
            project_id,
            text,
            kind=kind,
            shot_id=shot_id,
            scope="shot" if shot_id else "project",
        )

    hits_top1 = 0
    hits_top3 = 0
    latencies: list[float] = []
    token_costs: list[int] = []
    misses: list[dict] = []
    for query, expected, shot_id in TASKS:
        started = time.perf_counter()
        pack = agent_memory.context_pack(
            session, project_id, query, budget_tokens=400, shot_id=shot_id
        )
        latencies.append((time.perf_counter() - started) * 1000)
        texts = [row["text"] for row in pack["entries"]]
        token_costs.append(pack["used_tokens"])
        if texts and expected.lower() in texts[0].lower():
            hits_top1 += 1
        if any(expected.lower() in text.lower() for text in texts[:3]):
            hits_top3 += 1
        else:
            misses.append({"query": query, "expected": expected, "got": texts[:3]})

    latencies.sort()

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
        return round(values[index], 2)

    total = len(TASKS)
    return {
        "tasks": total,
        "top1_accuracy": round(hits_top1 / total, 3),
        "top3_accuracy": round(hits_top3 / total, 3),
        "avg_tokens": round(sum(token_costs) / total, 1),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-top3", type=float, default=0.8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_suite()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"agent memory eval: {report['tasks']} tasks · "
            f"top-1 {report['top1_accuracy']:.0%} · top-3 {report['top3_accuracy']:.0%} · "
            f"avg {report['avg_tokens']} tokens · "
            f"p50 {report['latency_ms_p50']}ms · p95 {report['latency_ms_p95']}ms"
        )
        for miss in report["misses"]:
            print(f"  miss: {miss['query']!r} wanted {miss['expected']!r}, got {miss['got']}")

    if report["top3_accuracy"] < args.min_top3:
        print(
            f"::error::top-3 accuracy {report['top3_accuracy']:.0%} below {args.min_top3:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
