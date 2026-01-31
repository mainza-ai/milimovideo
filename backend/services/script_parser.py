import re
from typing import List, Optional
from pydantic import BaseModel

class ParsedShot(BaseModel):
    action: str
    dialogue: Optional[str] = None
    character: Optional[str] = None # The speaker

class ParsedScene(BaseModel):
    name: str
    shots: List[ParsedShot]
    content: str # Raw text of the scene

class ScriptParser:
    def __init__(self):
        # Regex for Sluglines (INT. / EXT.)
        self.scene_heading_regex = re.compile(r'^\s*(?:INT\.|EXT\.|INT\./EXT\.|I/E)(?:[\s\.]).*$', re.MULTILINE | re.IGNORECASE)
        # Regex for Character Names (All Caps, indented or centered usually, but we check for ALL CAPS less than N chars)
        self.character_regex = re.compile(r'^\s*[A-Z][A-Z0-9\s\.]+\s*$')

    def parse_script(self, text: str) -> List[ParsedScene]:
        """
        Parses raw script text into a list of Scenes, each containing a list of Shot objects.
        """
        lines = text.split('\n')
        scenes = []
        current_scene = None
        current_buffer = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if line is a Scene Heading
            if self.scene_heading_regex.match(stripped):
                # Save previous scene if exists
                if current_scene:
                    current_scene.shots = self._parse_buffer_to_shots(current_buffer)
                    current_scene.content = "\n".join(current_buffer)
                    scenes.append(current_scene)
                
                # Start new scene
                current_scene = ParsedScene(name=stripped, shots=[], content="")
                current_buffer = []
            else:
                # Accumulate content for current scene
                if current_scene:
                    current_buffer.append(stripped)
        
        # Capture last scene
        if current_scene:
            current_scene.shots = self._parse_buffer_to_shots(current_buffer)
            current_scene.content = "\n".join(current_buffer)
            scenes.append(current_scene)
            
        return scenes

    def _parse_buffer_to_shots(self, buffer: List[str]) -> List[ParsedShot]:
        """
        Converts a list of lines into ParsedShots.
        Attempts to group Character + Dialogue, or Action blocks.
        """
        shots = []
        i = 0
        while i < len(buffer):
            line = buffer[i]
            
            # Check for Character Name (Dialogue Start)
            # Heuristic: Uppercase, relatively short, previous line was distinct?
            if self.character_regex.match(line) and len(line) < 30:
                character = line
                dialogue_lines = []
                i += 1
                while i < len(buffer):
                    next_line = buffer[i]
                    # Stop if next line is another character or empty (if we had empty lines)
                    # But buffer is stripped lines. 
                    # Stop if next line looks like a character or a new scene (already handled)
                    if self.character_regex.match(next_line) and len(next_line) < 30:
                        break
                    dialogue_lines.append(next_line)
                    i += 1
                
                # Create Dialogue Shot
                # For "shot", we usually need visual action. 
                # Dialogue implies: "Shot of [Character] talking".
                action_desc = f"Close up of {character} speaking."
                shots.append(ParsedShot(
                    action=action_desc,
                    dialogue=" ".join(dialogue_lines),
                    character=character
                ))
            else:
                # Assume Action
                shots.append(ParsedShot(action=line))
                i += 1
            
        return shots

script_parser = ScriptParser()
