"""
Element Matcher — Production-grade entity resolution for storyboards.

Matches project elements (characters, locations, objects) against parsed
script data using multi-signal analysis: name matching, trigger word
detection, action text scanning, and scene heading analysis.

Each match carries a confidence score (0.0–1.0) and source attribution.
"""

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ElementMatch:
    """A single element-to-shot match with provenance."""
    element_id: str
    element_name: str
    element_type: str          # character | location | object
    image_url: Optional[str]   # For frontend thumbnail display
    trigger_word: str
    confidence: float          # 0.0–1.0
    match_source: str          # Where the match was found

    def to_dict(self) -> dict:
        return asdict(self)


# ── Matching Thresholds ──────────────────────────────────────────────

MIN_CONFIDENCE = 0.4           # Below this, discard
EXACT_NAME_CONFIDENCE = 1.0
TRIGGER_WORD_CONFIDENCE = 0.95
PARTIAL_NAME_CONFIDENCE = 0.8
ACTION_NAME_CONFIDENCE = 0.75
SCENE_HEADING_CONFIDENCE = 0.85
DIALOGUE_MENTION_CONFIDENCE = 0.7
DESCRIPTION_KEYWORD_CONFIDENCE = 0.5


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase alpha tokens (2+ chars)."""
    return {w.lower() for w in re.findall(r'[a-zA-Z]{2,}', text)}


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace."""
    return text.strip().lower()


def _build_element_index(elements: list) -> list[dict]:
    """
    Pre-process elements into a uniform index for matching.
    Handles both SQLModel objects (with attributes) and dicts.
    """
    index = []
    for el in elements:
        # Support both dict-like and object access
        if hasattr(el, '__dict__') and not isinstance(el, dict):
            # SQLModel object
            entry = {
                "id": el.id,
                "name": el.name,
                "type": el.type,
                "trigger_word": el.trigger_word or "",
                "description": el.description or "",
                "image_path": getattr(el, "image_path", None),
            }
        else:
            entry = {
                "id": el.get("id", ""),
                "name": el.get("name", ""),
                "type": el.get("type", ""),
                "trigger_word": el.get("trigger_word", ""),
                "description": el.get("description", ""),
                "image_path": el.get("image_path", None),
            }

        # Pre-compute matching keys
        entry["name_lower"] = _normalize(entry["name"])
        entry["name_tokens"] = _tokenize(entry["name"])
        entry["trigger_lower"] = _normalize(entry["trigger_word"]).lstrip("@")
        entry["desc_tokens"] = _tokenize(entry["description"])

        index.append(entry)

    return index


def _match_shot(
    shot: dict,
    elements: list[dict],
    scene_name: str = "",
) -> list[ElementMatch]:
    """
    Match a single shot against all elements.
    Returns deduplicated matches sorted by confidence (highest first).
    """
    matches: dict[str, ElementMatch] = {}  # element_id -> best match

    character = shot.get("character") or ""
    action = shot.get("action") or ""
    dialogue = shot.get("dialogue") or ""

    char_lower = _normalize(character)
    action_lower = _normalize(action)
    dialogue_lower = _normalize(dialogue)
    scene_lower = _normalize(scene_name)

    action_tokens = _tokenize(action)
    dialogue_tokens = _tokenize(dialogue)

    for el in elements:
        best_confidence = 0.0
        best_source = ""

        # ── Signal 1: Character field exact match ────────────────
        if char_lower and char_lower == el["name_lower"]:
            best_confidence = EXACT_NAME_CONFIDENCE
            best_source = "character_field_exact"

        # ── Signal 2: Trigger word in character field ────────────
        if char_lower and el["trigger_lower"]:
            if el["trigger_lower"] == char_lower or el["trigger_lower"] in char_lower:
                if TRIGGER_WORD_CONFIDENCE > best_confidence:
                    best_confidence = TRIGGER_WORD_CONFIDENCE
                    best_source = "character_field_trigger"

        # ── Signal 3: Character field partial/token match ────────
        if char_lower and el["name_tokens"] and best_confidence < PARTIAL_NAME_CONFIDENCE:
            # Check if all name tokens appear in the character field
            char_tokens = _tokenize(character)
            overlap = el["name_tokens"] & char_tokens
            if overlap and len(overlap) >= len(el["name_tokens"]):
                best_confidence = PARTIAL_NAME_CONFIDENCE
                best_source = "character_field_partial"

        # ── Signal 4: Trigger word in action text ────────────────
        if el["trigger_lower"] and el["trigger_lower"] in action_lower:
            if TRIGGER_WORD_CONFIDENCE > best_confidence:
                best_confidence = TRIGGER_WORD_CONFIDENCE
                best_source = "action_trigger_word"

        # Full trigger with @ prefix
        full_trigger = f"@{el['trigger_lower']}"
        if full_trigger in action_lower:
            if TRIGGER_WORD_CONFIDENCE > best_confidence:
                best_confidence = TRIGGER_WORD_CONFIDENCE
                best_source = "action_trigger_word_prefixed"

        # ── Signal 5: Element name in action text ────────────────
        if el["name_lower"] and el["name_lower"] in action_lower:
            if ACTION_NAME_CONFIDENCE > best_confidence:
                best_confidence = ACTION_NAME_CONFIDENCE
                best_source = "action_name_mention"

        # Token-level name matching in action (for multi-word names)
        if el["name_tokens"] and best_confidence < ACTION_NAME_CONFIDENCE:
            overlap = el["name_tokens"] & action_tokens
            if overlap and len(overlap) >= len(el["name_tokens"]):
                best_confidence = ACTION_NAME_CONFIDENCE
                best_source = "action_name_tokens"

        # ── Signal 6: Scene heading match (locations) ────────────
        if scene_lower and el["type"] == "location":
            if el["name_lower"] in scene_lower:
                if SCENE_HEADING_CONFIDENCE > best_confidence:
                    best_confidence = SCENE_HEADING_CONFIDENCE
                    best_source = "scene_heading"

        # ── Signal 7: Dialogue mentions ──────────────────────────
        if dialogue_lower and el["name_lower"] in dialogue_lower:
            if DIALOGUE_MENTION_CONFIDENCE > best_confidence:
                best_confidence = DIALOGUE_MENTION_CONFIDENCE
                best_source = "dialogue_mention"

        # ── Signal 8: Description keyword overlap ────────────────
        if el["desc_tokens"] and best_confidence < DESCRIPTION_KEYWORD_CONFIDENCE:
            combined_tokens = action_tokens | dialogue_tokens
            if combined_tokens:
                overlap = el["desc_tokens"] & combined_tokens
                # Require meaningful overlap (at least 3 shared keywords)
                if len(overlap) >= 3:
                    ratio = len(overlap) / len(el["desc_tokens"])
                    score = DESCRIPTION_KEYWORD_CONFIDENCE * min(ratio, 1.0)
                    if score > best_confidence:
                        best_confidence = score
                        best_source = "description_keywords"

        # ── Record match if above threshold ──────────────────────
        if best_confidence >= MIN_CONFIDENCE:
            existing = matches.get(el["id"])
            if not existing or best_confidence > existing.confidence:
                matches[el["id"]] = ElementMatch(
                    element_id=el["id"],
                    element_name=el["name"],
                    element_type=el["type"],
                    image_url=el["image_path"],
                    trigger_word=el.get("trigger_word", ""),
                    confidence=round(best_confidence, 2),
                    match_source=best_source,
                )

    # Sort by confidence descending
    return sorted(matches.values(), key=lambda m: m.confidence, reverse=True)


def match_elements(scenes: list[dict], elements: list) -> list[dict]:
    """
    Annotate each shot in each scene with matched elements.

    Args:
        scenes: Parsed scene list [{name, shots: [{action, character, ...}]}]
        elements: Project elements (SQLModel objects or dicts)

    Returns:
        Same scene list with `matched_elements` added to each shot.
    """
    if not elements:
        logger.debug("No elements to match against")
        return scenes

    element_index = _build_element_index(elements)
    logger.info(f"Element matcher: {len(element_index)} elements indexed")

    total_matches = 0

    for scene in scenes:
        scene_name = scene.get("name", "")
        for shot in scene.get("shots", []):
            shot_matches = _match_shot(shot, element_index, scene_name)
            shot["matched_elements"] = [m.to_dict() for m in shot_matches]
            total_matches += len(shot_matches)

    logger.info(f"Element matcher: {total_matches} total matches across all shots")
    return scenes


def build_element_manifest(elements: list) -> str:
    """
    Build a human-readable element manifest for AI prompt injection.

    Returns a formatted string listing all available elements by type
    that can be appended to the AI system prompt.
    """
    if not elements:
        return ""

    index = _build_element_index(elements)

    by_type: dict[str, list[dict]] = {}
    for el in index:
        t = el["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(el)

    lines = ["\n### Available Project Elements"]
    type_labels = {
        "character": "Characters",
        "location": "Locations",
        "object": "Objects/Props",
    }

    for el_type, label in type_labels.items():
        group = by_type.get(el_type, [])
        if group:
            entries = []
            for el in group:
                trigger = el["trigger_word"] if el["trigger_word"] else ""
                desc_snippet = el["description"][:60] if el["description"] else ""
                entry = f"{el['name']}"
                if trigger:
                    entry += f" ({trigger})"
                if desc_snippet:
                    entry += f" — {desc_snippet}"
                entries.append(entry)
            lines.append(f"**{label}**: {', '.join(entries)}")

    lines.append("")
    lines.append(
        "When you detect these characters, locations, or objects in the script, "
        "use their EXACT names (as listed above) in the character field. "
        "For locations, reference them in the scene name."
    )

    return "\n".join(lines)
