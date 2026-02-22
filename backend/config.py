import os
import sys

# Base Directories
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
PROJECTS_DIR = os.path.join(BACKEND_DIR, "projects")
GENERATED_DIR = os.path.join(BACKEND_DIR, "generated") # Legacy Support

# LTX-2 Paths
LTX_DIR = os.path.join(PROJECT_ROOT, "LTX-2")
LTX_CORE_DIR = os.path.join(LTX_DIR, "packages/ltx-core/src")
LTX_PIPELINES_DIR = os.path.join(LTX_DIR, "packages/ltx-pipelines/src")

# Model Paths (Can be updated to point to specific weights)
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
FLUX_WEIGHTS_PATH = os.path.join(MODELS_DIR, "flux2") # Updated path to directory
SAM_WEIGHTS_PATH = os.path.join(MODELS_DIR, "sam3", "sam3.pt")

# IP-Adapter Paths (Visual Conditioning)
FLUX_IP_ADAPTER_PATH = os.path.join(MODELS_DIR, "flux-ip-adapter.safetensors") # Placeholder
LTX_IP_ADAPTER_PATH = os.path.join(MODELS_DIR, "ltx-ip-adapter.safetensors")   # Placeholder

# Service Ports
API_PORT = 8000
SAM_SERVICE_PORT = 8001

# Database
DATABASE_URL = f"sqlite:///{os.path.join(BACKEND_DIR, 'milimovideo.db')}"

# Generation Defaults
DEFAULT_RESOLUTION_W = 768
DEFAULT_RESOLUTION_H = 512
DEFAULT_FPS = 25
DEFAULT_SEED = 42
DEFAULT_NUM_FRAMES = 121       # LTX-2 default chunk size (was hardcoded)
NATIVE_MAX_FRAMES = 121        # LTX-2 native maximum (~20.2s at 25fps)
DEFAULT_OVERLAP_FRAMES = 25   # Frames overlapped between chained chunks (must be 1+8k for native VAE encoding)
DEFAULT_THUMBNAIL_W = 512     # Concept art thumbnail width
DEFAULT_THUMBNAIL_H = 320     # Concept art thumbnail height

def setup_paths():
    """Ensure necessary paths are in sys.path"""
    if LTX_CORE_DIR not in sys.path:
        sys.path.append(LTX_CORE_DIR)
    if LTX_PIPELINES_DIR not in sys.path:
        sys.path.insert(0, LTX_PIPELINES_DIR)
    if LTX_CORE_DIR not in sys.path:
        sys.path.insert(0, LTX_CORE_DIR)

# Ensure directories exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# MPS Memory Safety
# Maximum safe generation resolution for Apple MPS with 19B float32 model
# The two-stage pipeline runs Stage 1 at half-res and Stage 2 at full-res.
# Attention memory scales quadratically with token count (H*W*Frames).
MPS_MAX_RESOLUTION_W = 1280
MPS_MAX_RESOLUTION_H = 768
MPS_MAX_PIXELS = MPS_MAX_RESOLUTION_W * MPS_MAX_RESOLUTION_H  # ~983K pixels

PRESETS = {
    "Standard": {"width": 768, "height": 512},
    "Widescreen": {"width": 1024, "height": 576},
    "HD": {"width": 1280, "height": 704},
    "Portrait": {"width": 512, "height": 768},
}

FILTER_PRESETS = [
    {"name": "Cinematic", "prompt": "cinematic lighting, shallow depth of field, 35mm film grain, high budget movie"},
    {"name": "Cyberpunk", "prompt": "neon lights, rain, wet streets, futuristic city, high tech, blue and pink lighting"},
    {"name": "Anime", "prompt": "studio ghibli style, cel shaded, vibrant colors, detailed background"},
    {"name": "Vintage", "prompt": "1950s footage, black and white, film grain, scratches, flickering"},
]

# ── LLM Configuration ─────────────────────────────────────────────
# Provider: "gemma" (built-in LTX text encoder) or "ollama" (local LLM)
LLM_PROVIDER = os.environ.get("MILIMO_LLM_PROVIDER", "gemma")
OLLAMA_BASE_URL = os.environ.get("MILIMO_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("MILIMO_OLLAMA_MODEL", "llama3.1")
# "0" = unload model immediately after use (saves ~49GB during generation)
# "5m" = keep loaded 5 minutes (Ollama default, faster re-prompts but wastes RAM)
OLLAMA_KEEP_ALIVE = os.environ.get("MILIMO_OLLAMA_KEEP_ALIVE", "0")

