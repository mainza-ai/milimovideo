"""Local role fixtures for the eval harness.

These exercise the real contract Milimo depends on: no JSON wrappers reaching
the image model, action-shaped video prompts, and parseable scene lists. They
are deterministic and local-only.
"""

from __future__ import annotations

VIDEO_FIXTURES = [
    {"id": "noir-alley", "prompt": "Detective Mara steps into a rain-soaked alley, neon flickering above her."},
    {"id": "lighthouse", "prompt": "A lighthouse in a storm, waves crashing over the rocks."},
    {"id": "studio-night", "prompt": "A filmmaker edits alone at night, the timeline glowing."},
    {"id": "rooftop-dawn", "prompt": "Mara watches the city wake up from a rooftop."},
]

IMAGE_FIXTURES = [
    {"id": "portrait-rain", "prompt": "A portrait of a detective standing in the rain."},
    {"id": "neon-alley", "prompt": "A neon alley at midnight, wet cobblestones."},
    {"id": "desert-car", "prompt": "A rusted car in the desert at golden hour."},
]

SCRIPT_FIXTURES = [
    {
        "id": "two-scenes",
        "text": (
            "INT. STUDIO - NIGHT\n"
            "Mara edits alone. The timeline glows.\n"
            "MARA\nWe are not done.\n\n"
            "EXT. ROOFTOP - DAWN\n"
            "Mara watches the city wake up."
        ),
    },
    {
        "id": "single-scene",
        "text": (
            "INT. KITCHEN - MORNING\n"
            "A kettle boils. Sam pours two cups and slides one across the table."
        ),
    },
]

FIXTURES = {
    "prompt_video": VIDEO_FIXTURES,
    "prompt_image": IMAGE_FIXTURES,
    "script_parse": SCRIPT_FIXTURES,
}

DEFAULT_EVAL_ROLES = ["prompt_video", "prompt_image", "script_parse"]
FIXTURE_VERSION = "2026-11-17.1"
