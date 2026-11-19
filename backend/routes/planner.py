"""Deterministic, local plan previews for the studio command dock.

The planner never mutates project state and never calls a model. It turns
explicit directives into a diff that the client can show before approval, then
the client applies those patches through the undoable document store. Generation
stays behind the explicit approve/spend boundary.
"""

import re
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from database import get_session, Project, Shot
from schemas import PlanEstimate, PlanRequest, PlanResponse, PlanStep

router = APIRouter(tags=["planner"])

# The app's canonical ShotType union is close_up | medium | wide | establishing
# | insert | tracking. Accept friendly aliases but emit canonical values.
SHOT_TYPE_ALIASES = {
    "close-up": "close_up",
    "close up": "close_up",
    "closeup": "close_up",
    "extreme close-up": "close_up",
    "extreme close up": "close_up",
    "medium": "medium",
    "medium shot": "medium",
    "two-shot": "medium",
    "two shot": "medium",
    "over-the-shoulder": "medium",
    "over the shoulder": "medium",
    "pov": "medium",
    "wide": "wide",
    "wide shot": "wide",
    "establishing": "establishing",
    "establishing shot": "establishing",
    "aerial": "establishing",
    "insert": "insert",
    "insert shot": "insert",
    "tracking": "tracking",
    "tracking shot": "tracking",
}

SEED_RE = re.compile(r"\bseed\s*(?:to|=|:)?\s*(\d{1,10})\b", re.IGNORECASE)
FRAMES_RE = re.compile(r"\b(?:frames?|length)\s*(?:to|=|:)?\s*(\d{1,4})\b", re.IGNORECASE)
TYPE_RE = re.compile(
    r"\b(?:shot\s*type|type|make\s+it\s+a|change\s+to)\s*[:=]?\s*([a-z][a-z\- ]{2,24})",
    re.IGNORECASE,
)
PROMPT_RE = re.compile(r"\bprompt\s*[:=]\s*(.+)$", re.IGNORECASE | re.DOTALL)
NEGATIVE_RE = re.compile(r"\bnegative(?:\s*prompt)?\s*[:=]\s*(.+)$", re.IGNORECASE | re.DOTALL)
APPEND_RE = re.compile(r"\b(?:add|append|make\s+it|more)\s+(.+)$", re.IGNORECASE | re.DOTALL)


def _shot_or_404(session: Session, project: Project, shot_id: Optional[str]) -> Optional[Shot]:
    if not shot_id:
        return None
    shot = session.get(Shot, shot_id)
    if not shot or shot.project_id != project.id:
        raise HTTPException(status_code=404, detail="Shot not found in this project")
    return shot


def _step(kind: str, label: str, target: str, before, after, detail: Optional[str] = None) -> PlanStep:
    return PlanStep(kind=kind, label=label, target=target, before=before, after=after, detail=detail)


def _direct_steps(shot: Shot, text: str) -> Tuple[List[PlanStep], List[str]]:
    steps: List[PlanStep] = []
    warnings: List[str] = []
    working = text
    target = shot.id

    match = PROMPT_RE.search(working)
    if match:
        value = match.group(1).strip()
        steps.append(_step("set_prompt", "Replace prompt", target, shot.prompt, value))
        working = working[:match.start()] + working[match.end():]

    match = NEGATIVE_RE.search(working)
    if match:
        value = match.group(1).strip()
        steps.append(_step(
            "set_negative_prompt", "Replace negative prompt", target,
            shot.negative_prompt, value,
        ))
        working = working[:match.start()] + working[match.end():]

    match = SEED_RE.search(working)
    if match:
        value = int(match.group(1))
        steps.append(_step("set_seed", f"Lock seed to {value}", target, shot.seed, value))

    match = FRAMES_RE.search(working)
    if match:
        value = max(1, min(2000, int(match.group(1))))
        steps.append(_step("set_frames", f"Set length to {value} frames", target, shot.num_frames, value))

    match = TYPE_RE.search(working)
    if match:
        value = match.group(1).strip().lower().rstrip(".,;")
        canonical = SHOT_TYPE_ALIASES.get(value)
        if canonical:
            steps.append(_step(
                "set_shot_type", f"Shot type: {canonical}", target, shot.shot_type, canonical,
            ))
        else:
            warnings.append(f"Unrecognized shot type '{value}'.")

    if not steps:
        match = APPEND_RE.search(working)
        if match:
            phrase = match.group(1).strip().rstrip(".,;")
            if phrase:
                after = f"{(shot.prompt or '').rstrip('. ')}, {phrase}".strip(", ")
                steps.append(_step(
                    "set_prompt", "Append to prompt", target, shot.prompt, after,
                    detail=f"Adds: {phrase}",
                ))

    if not steps:
        warnings.append(
            "No deterministic edit matched. Use explicit directives "
            "(prompt:, negative:, seed N, frames N, type close-up, add ...)."
        )
    return steps, warnings


def _estimate(project: Project, shot: Shot, frames: int, width: int, height: int) -> PlanEstimate:
    fps = shot.fps or project.fps or 25
    baseline = (768 * 512) * 121
    workload = max(0.08, ((width * height) * frames) / baseline)
    chunks = max(1, (frames + 120) // 121)
    model = shot.pipeline_override or "auto (LTX-2)"
    return PlanEstimate(
        model=model,
        gpu_seconds=round(8 + 22 * workload, 1),
        vram_mb=int(min(30000, 11000 + 6000 * workload)),
        note=(
            f"Rough local estimate at {fps} fps, {chunks} chunk(s); "
            "the queue reports actual time."
        ),
    )


def _ask_answer(
    project: Project,
    shot: Optional[Shot],
    *,
    session: Session | None = None,
    query: str = "",
) -> str:
    total = len(project.shots)
    completed = sum(1 for item in project.shots if item.status == "completed")
    generating = sum(1 for item in project.shots if item.status == "generating")
    frames = sum(item.num_frames or 0 for item in project.shots)
    fps = project.fps or 25
    minutes, seconds = divmod(round(frames / max(1, fps)), 60)
    base = (
        f"{project.name}: {total} shots, {completed} with takes, {generating} generating. "
        f"Cut length about {minutes}m {seconds:02d}s at {project.resolution_w}x"
        f"{project.resolution_h}, {fps} fps."
    )
    if shot:
        base += (
            f" Selected shot {shot.id}: seed {shot.seed}, {shot.num_frames} frames, "
            f"model {shot.pipeline_override or 'auto'}."
        )
    if session is not None:
        # Retrieved project memory, packed to a small token budget. Quarantined
        # entries are excluded by the packer, so agent/imported text cannot
        # steer an answer before a human approves it.
        try:
            from agent_memory import context_pack

            pack = context_pack(
                session,
                project.id,
                query or project.name,
                budget_tokens=220,
                shot_id=shot.id if shot else None,
            )
        except Exception:
            pack = {"entries": []}
        if pack.get("entries"):
            facts = "; ".join(entry["text"] for entry in pack["entries"][:4])
            base += f" Project memory: {facts}"
    return base


@router.post("/projects/{project_id}/plan", response_model=PlanResponse)
def create_plan(
    project_id: str,
    req: PlanRequest,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    text = (req.text or "").strip()
    shot = _shot_or_404(session, project, req.shot_id)
    mode = req.mode if req.mode in {"ask", "direct", "generate", "find"} else "direct"

    if mode == "ask":
        answer = _ask_answer(project, shot, session=session, query=text)
        return PlanResponse(
            mode=mode,
            summary=answer,
            steps=[PlanStep(kind="answer", label="Answer", detail=answer)],
            shot_id=shot.id if shot else None,
        )

    if mode == "find":
        return PlanResponse(
            mode=mode,
            summary=f"Filter the board for “{text}”." if text else "Type a search term.",
            steps=[PlanStep(kind="search", label=f"Search: {text}", after=text)],
            shot_id=shot.id if shot else None,
        )

    if not shot:
        raise HTTPException(status_code=400, detail="Select a shot before planning an edit")

    if mode == "generate":
        if not text:
            raise HTTPException(status_code=400, detail="Describe the take to generate")
        frames = shot.num_frames or 121
        estimate = _estimate(project, shot, frames, shot.width, shot.height)
        return PlanResponse(
            mode=mode,
            summary=f"Generate a new take for shot {shot.id}.",
            steps=[PlanStep(
                kind="generate",
                label="Generate take",
                target=shot.id,
                before=shot.video_url,
                after=text,
                detail=f"seed {shot.seed} · {frames} frames · {shot.width}x{shot.height}",
            )],
            estimate=estimate,
            shot_id=shot.id,
        )

    steps, warnings = _direct_steps(shot, text)
    summary = (
        f"{len(steps)} edit(s) on shot {shot.id}." if steps
        else "No edits planned."
    )
    return PlanResponse(
        mode=mode,
        summary=summary,
        steps=steps,
        warnings=warnings,
        shot_id=shot.id,
    )
