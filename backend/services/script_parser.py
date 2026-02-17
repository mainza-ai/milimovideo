import re
from typing import List, Optional, Literal
from pydantic import BaseModel

SHOT_TYPES = ['close_up', 'medium', 'wide', 'establishing', 'insert', 'tracking']

DIALOGUE_SHOT_PATTERNS = [
    "Close up of {character} speaking.",
    "Medium shot of {character} delivering their lines.",
    "Over the shoulder of {character} as they speak.",
    "Two-shot framing {character} in conversation.",
    "Close up reaction shot of {character}.",
    "Tracking shot following {character} as they talk.",
]

class ParsedShot(BaseModel):
    action: str
    dialogue: Optional[str] = None
    character: Optional[str] = None
    shot_type: Optional[str] = None  # close_up, medium, wide, establishing, insert, tracking

class ParsedScene(BaseModel):
    name: str
    shots: List[ParsedShot]
    content: str  # Raw text of the scene

class ScriptParser:
    def __init__(self):
        # Regex for Sluglines (INT. / EXT.)
        self.scene_heading_regex = re.compile(
            r'^\s*(?:INT\.|EXT\.|INT\./EXT\.|I/E)(?:[\s\.]).*$',
            re.MULTILINE | re.IGNORECASE
        )
        # Regex for Character Names (All Caps, < 30 chars)
        self.character_regex = re.compile(r'^\s*[A-Z][A-Z0-9\s\.]+\s*$')
        # Regex for numbered list items: "1.", "1)", "Shot 1:", "#1"
        self.numbered_regex = re.compile(
            r'^\s*(?:(?:Shot\s+)?#?\d+[\.\):\-])\s*(.+)',
            re.IGNORECASE
        )
        # Counter for round-robin dialogue shot variety
        self._dialogue_counter = 0

    def parse_script(self, text: str, parse_mode: str = "auto") -> List[ParsedScene]:
        """
        Parses raw script text into a list of Scenes, each containing Shots.

        parse_mode:
            "auto"       — detect format automatically
            "screenplay"  — force INT./EXT. heading detection
            "freeform"    — split by paragraph breaks
            "numbered"    — detect numbered list items
        """
        if not text or not text.strip():
            return []

        if parse_mode == "auto":
            parse_mode = self._detect_mode(text)

        if parse_mode == "screenplay":
            return self._parse_screenplay(text)
        elif parse_mode == "numbered":
            return self._parse_numbered(text)
        else:  # freeform
            return self._parse_freeform(text)

    def _detect_mode(self, text: str) -> str:
        """Auto-detect the best parsing mode for the input text."""
        # Check for screenplay headings
        if self.scene_heading_regex.search(text):
            return "screenplay"

        # Check for numbered list patterns
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        numbered_count = sum(1 for l in lines if self.numbered_regex.match(l))
        if numbered_count >= 2 and numbered_count >= len(lines) * 0.5:
            return "numbered"

        # Default: free-form
        return "freeform"

    # ── Screenplay Mode ──────────────────────────────────────────────

    def _parse_screenplay(self, text: str) -> List[ParsedScene]:
        """Original screenplay parser: INT./EXT. headings → scenes, ALL CAPS → characters."""
        lines = text.split('\n')
        scenes: List[ParsedScene] = []
        current_scene: Optional[ParsedScene] = None
        current_buffer: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if self.scene_heading_regex.match(stripped):
                if current_scene:
                    current_scene.shots = self._parse_buffer_to_shots(current_buffer)
                    current_scene.content = "\n".join(current_buffer)
                    scenes.append(current_scene)

                current_scene = ParsedScene(name=stripped, shots=[], content="")
                current_buffer = []
            else:
                if current_scene:
                    current_buffer.append(stripped)

        # Capture last scene
        if current_scene:
            current_scene.shots = self._parse_buffer_to_shots(current_buffer)
            current_scene.content = "\n".join(current_buffer)
            scenes.append(current_scene)

        # If screenplay headings were expected but nothing matched, fall through to freeform
        if not scenes:
            return self._parse_freeform(text)

        return scenes

    # ── Free-form Mode ────────────────────────────────────────────────

    def _parse_freeform(self, text: str) -> List[ParsedScene]:
        """Split by paragraph breaks → individual shots within a single Scene 1."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        shots: List[ParsedShot] = []
        for i, para in enumerate(paragraphs):
            # Merge multi-line paragraphs into one action
            action = ' '.join(para.split())
            shot_type = self._infer_shot_type(action, i, len(paragraphs))
            shots.append(ParsedShot(
                action=action,
                shot_type=shot_type
            ))

        return [ParsedScene(
            name="Scene 1",
            shots=shots,
            content=text.strip()
        )]

    # ── Numbered List Mode ────────────────────────────────────────────

    def _parse_numbered(self, text: str) -> List[ParsedScene]:
        """Detect 1., Shot 1:, #1 patterns → individual shots."""
        lines = text.split('\n')
        shots: List[ParsedShot] = []
        current_item_lines: List[str] = []
        in_numbered = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = self.numbered_regex.match(stripped)
            if match:
                # Flush previous item
                if current_item_lines:
                    action = ' '.join(current_item_lines)
                    shots.append(ParsedShot(
                        action=action,
                        shot_type=self._infer_shot_type(action, len(shots), -1)
                    ))
                current_item_lines = [match.group(1).strip()]
                in_numbered = True
            elif in_numbered:
                # Continuation line of the current numbered item
                current_item_lines.append(stripped)
            else:
                # Non-numbered line before any numbered item — treat as standalone shot
                shots.append(ParsedShot(
                    action=stripped,
                    shot_type=self._infer_shot_type(stripped, len(shots), -1)
                ))

        # Flush last item
        if current_item_lines:
            action = ' '.join(current_item_lines)
            shots.append(ParsedShot(
                action=action,
                shot_type=self._infer_shot_type(action, len(shots), -1)
            ))

        if not shots:
            return self._parse_freeform(text)

        return [ParsedScene(
            name="Scene 1",
            shots=shots,
            content=text.strip()
        )]

    # ── Buffer → Shots (screenplay character/dialogue grouping) ──────

    def _parse_buffer_to_shots(self, buffer: List[str]) -> List[ParsedShot]:
        """Converts screenplay scene buffer into ParsedShots with character/dialogue grouping."""
        shots: List[ParsedShot] = []
        i = 0
        while i < len(buffer):
            line = buffer[i]

            if self.character_regex.match(line) and len(line) < 30:
                character = line
                dialogue_lines: List[str] = []
                i += 1
                while i < len(buffer):
                    next_line = buffer[i]
                    if self.character_regex.match(next_line) and len(next_line) < 30:
                        break
                    dialogue_lines.append(next_line)
                    i += 1

                # Round-robin dialogue shot descriptions
                pattern = DIALOGUE_SHOT_PATTERNS[self._dialogue_counter % len(DIALOGUE_SHOT_PATTERNS)]
                action_desc = pattern.format(character=character)
                self._dialogue_counter += 1

                # Infer shot type from the pattern used
                shot_type = self._shot_type_from_dialogue_pattern(self._dialogue_counter - 1)

                shots.append(ParsedShot(
                    action=action_desc,
                    dialogue=" ".join(dialogue_lines),
                    character=character,
                    shot_type=shot_type
                ))
            else:
                shot_type = self._infer_shot_type(line, len(shots), -1)
                shots.append(ParsedShot(action=line, shot_type=shot_type))
                i += 1

        return shots

    # ── Shot Type Inference ───────────────────────────────────────────

    def _infer_shot_type(self, action: str, index: int, total: int) -> str:
        """Infer a reasonable shot type from the action text and position."""
        action_lower = action.lower()

        # Keyword-based inference
        if any(kw in action_lower for kw in ['establishing', 'skyline', 'landscape', 'aerial', 'exterior']):
            return 'establishing'
        if any(kw in action_lower for kw in ['close up', 'closeup', 'close-up', 'detail', 'eye', 'hand']):
            return 'close_up'
        if any(kw in action_lower for kw in ['wide shot', 'wide angle', 'panorama', 'vista']):
            return 'wide'
        if any(kw in action_lower for kw in ['insert', 'cutaway', 'object', 'prop']):
            return 'insert'
        if any(kw in action_lower for kw in ['tracking', 'follow', 'chase', 'running', 'walking']):
            return 'tracking'

        # Position-based: first shot is often establishing, last is often wide
        if index == 0 and total > 2:
            return 'establishing'

        return 'medium'

    def _shot_type_from_dialogue_pattern(self, counter: int) -> str:
        """Map dialogue pattern index to shot type."""
        mapping = ['close_up', 'medium', 'medium', 'medium', 'close_up', 'tracking']
        return mapping[counter % len(mapping)]

script_parser = ScriptParser()
