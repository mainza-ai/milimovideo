"""
AI Storyboard — Gemma 3 powered script analysis.

Uses the Gemma text encoder's chat interface to intelligently parse
free-form text into structured scenes and shots with cinematic
action descriptions, character detection, and shot type suggestions.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────

AI_STORYBOARD_SYSTEM_PROMPT = """\
You are a professional storyboard artist and screenwriter. Given a block of text \
(a script, treatment, or free-form description), you will break it down into \
scenes and shots suitable for cinematic video generation.

### Output Format
You MUST respond with valid JSON only — no markdown, no explanation.
```json
[
  {
    "name": "Scene heading or summary",
    "shots": [
      {
        "action": "Visual action description for this shot",
        "dialogue": "Spoken dialogue (if any, or null)",
        "character": "Character name (if any, or null)",
        "shot_type": "one of: close_up, medium, wide, establishing, insert, tracking"
      }
    ]
  }
]
```

### Guidelines
- Each scene should have 1-8 shots.
- Action descriptions should be vivid and visual, suitable for AI video generation.
- Use varied shot types for cinematic interest.
- If no explicit scene breaks, create logical scenes from the narrative flow.
- If dialogue is present, pair it with a close_up or medium shot of the speaker.
- For transitions or establishing moments, use wide or establishing shots.
- Character names should be in TITLE CASE.
- Respond ONLY with the JSON array, nothing else."""


def ai_parse_script(
    text: str,
    text_encoder,
    max_new_tokens: int = 1024,
    seed: int = 42,
) -> list[dict]:
    """
    Use Gemma 3 to intelligently parse script text into scenes/shots.
    
    Args:
        text: Raw script or treatment text.
        text_encoder: A GemmaTextEncoderModelBase instance with model+processor loaded.
        max_new_tokens: Max tokens for Gemma to generate.
        seed: RNG seed for reproducibility.
    
    Returns:
        List of dicts matching ScriptParser output format:
        [{"name": str, "shots": [{"action": str, "dialogue": str|None, ...}]}]
    """
    messages = [
        {"role": "system", "content": AI_STORYBOARD_SYSTEM_PROMPT},
        {"role": "user", "content": f"Break this into a storyboard:\n\n{text}"},
    ]

    try:
        # Use the Gemma _enhance interface (chat completion)
        raw_output = text_encoder._enhance(
            messages=messages,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        logger.info(f"AI Parse raw output ({len(raw_output)} chars): {raw_output[:500]}")

        # Extract JSON from response (handle markdown fences)
        parsed = _extract_json(raw_output)

        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError("AI response did not produce a valid scene list")

        # Validate / sanitize structure
        return _validate_scenes(parsed)

    except Exception as e:
        logger.error(f"AI parse failed: {e}")
        raise


def _extract_json(text: str) -> list | dict:
    """Extract JSON from Gemma output, handling markdown fences and preamble."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from AI response: {text[:200]}")


def _validate_scenes(scenes: list[dict]) -> list[dict]:
    """Validate and sanitize AI-generated scene data."""
    valid_shot_types = {"close_up", "medium", "wide", "establishing", "insert", "tracking"}
    result = []

    for i, scene in enumerate(scenes):
        name = scene.get("name") or f"Scene {i + 1}"
        shots = scene.get("shots", [])

        valid_shots = []
        for shot in shots:
            action = shot.get("action") or shot.get("description") or ""
            dialogue = shot.get("dialogue")
            character = shot.get("character")
            shot_type = shot.get("shot_type", "medium")

            # Normalize shot_type
            if shot_type not in valid_shot_types:
                shot_type = "medium"

            # Ensure dialogue/character are str or None
            if dialogue and not isinstance(dialogue, str):
                dialogue = str(dialogue)
            if character and not isinstance(character, str):
                character = str(character)

            valid_shots.append({
                "action": action,
                "dialogue": dialogue if dialogue else None,
                "character": character if character else None,
                "shot_type": shot_type,
            })

        result.append({
            "name": name,
            "shots": valid_shots if valid_shots else [
                {"action": "A cinematic shot.", "dialogue": None, "character": None, "shot_type": "medium"}
            ],
        })

    return result
