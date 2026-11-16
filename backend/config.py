import os
import sys

# Base Directories
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
PROJECTS_DIR = os.environ.get("MILIMO_PROJECTS_DIR", os.path.join(BACKEND_DIR, "projects"))
GENERATED_DIR = os.path.join(BACKEND_DIR, "generated") # Legacy Support

# LTX-2 Paths
LTX_DIR = os.path.join(PROJECT_ROOT, "LTX-2")
LTX_CORE_DIR = os.path.join(LTX_DIR, "packages/ltx-core/src")
LTX_PIPELINES_DIR = os.path.join(LTX_DIR, "packages/ltx-pipelines/src")
FLUX_SRC_DIR = os.path.join(PROJECT_ROOT, "flux2", "src")

# Model Paths (Can be updated to point to specific weights)
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
FLUX_WEIGHTS_PATH = os.path.join(MODELS_DIR, "flux2") # Updated path to directory
SAM_WEIGHTS_PATH = os.path.join(MODELS_DIR, "sam3", "sam3.pt")

# Model Management
MODELS_MANIFEST_PATH = os.path.join(BACKEND_DIR, "models_manifest.json")
DOWNLOAD_TEMP_SUFFIX = ".downloading"
DOWNLOAD_PROGRESS_INTERVAL = 2.0  # seconds between SSE progress updates

# IP-Adapter Paths (Visual Conditioning)
FLUX_IP_ADAPTER_PATH = os.path.join(MODELS_DIR, "flux-ip-adapter.safetensors") # Placeholder
LTX_IP_ADAPTER_PATH = os.path.join(MODELS_DIR, "ltx-ip-adapter.safetensors")   # Placeholder

# Service Ports / exposure
API_PORT = int(os.environ.get("MILIMO_API_PORT", "8000"))
# Local-first default. Set MILIMO_HOST=0.0.0.0 only with MILIMO_API_TOKEN.
API_HOST = os.environ.get("MILIMO_HOST", "127.0.0.1")
SAM_SERVICE_PORT = int(os.environ.get("MILIMO_SAM_PORT", "8001"))

# Comma-separated allowlist. Empty means the safe local defaults.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "MILIMO_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173,"
        "http://tauri.localhost,https://tauri.localhost",
    ).split(",")
    if origin.strip()
]
# Optional bearer token. Required when binding to a non-loopback host.
API_TOKEN = os.environ.get("MILIMO_API_TOKEN", "").strip()

# Debug artifacts (request dumps, error dumps). Never on by default.
DEBUG = os.environ.get("MILIMO_DEBUG", "0") == "1"
SKIP_VRAM_CHECK = os.environ.get("MILIMO_SKIP_VRAM_CHECK", "0") == "1"
VRAM_ESTIMATE_VIDEO_GB = 40.0
VRAM_ESTIMATE_IMAGE_GB = 24.0

# Upload safety
MAX_UPLOAD_MB = int(os.environ.get("MILIMO_MAX_UPLOAD_MB", "2048"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    ".srt", ".vtt", ".json", ".txt", ".md",
}

# Database
DATABASE_URL = os.environ.get(
    "MILIMO_DATABASE_URL",
    f"sqlite:///{os.path.join(BACKEND_DIR, 'milimovideo.db')}",
)

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
    if FLUX_SRC_DIR not in sys.path:
        sys.path.insert(0, FLUX_SRC_DIR)

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

# Make the editable LTX/Flux checkouts importable even when the venv was built
# for a different repository location. server.py imports routes immediately,
# and routes import tasks -> ltx_pipelines, so this must run at import time.
setup_paths()

