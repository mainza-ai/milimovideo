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


PRESETS = {
    "Standard": {"width": 1280, "height": 720},
    "Cinematic": {"width": 1920, "height": 800},
    "4K": {"width": 3840, "height": 2160},
    "Social": {"width": 1080, "height": 1920}
}

FILTER_PRESETS = [
    {"name": "Cinematic", "prompt": "cinematic lighting, shallow depth of field, 35mm film grain, high budget movie"},
    {"name": "Cyberpunk", "prompt": "neon lights, rain, wet streets, futuristic city, high tech, blue and pink lighting"},
    {"name": "Anime", "prompt": "studio ghibli style, cel shaded, vibrant colors, detailed background"},
    {"name": "Vintage", "prompt": "1950s footage, black and white, film grain, scratches, flickering"},
]
