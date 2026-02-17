import sys
import os
import logging
import logging.handlers
import mimetypes
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config

# ── Structured Logging Setup ──────────────────────────────────────
# File handler: JSON-structured for machine parsing
# Console handler: Human-readable for development

class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production log files."""
    def format(self, record):
        import json, time
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "logger": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

# File handler: structured JSON (rotating, max 10MB, keep 3 backups)
file_handler = logging.handlers.RotatingFileHandler(
    "server.log", maxBytes=10*1024*1024, backupCount=3
)
file_handler.setFormatter(StructuredFormatter())
file_handler.setLevel(logging.INFO)

# Console handler: human-readable  
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(config.PROJECTS_DIR, exist_ok=True)
legacy_upload_dir = os.path.join(config.BACKEND_DIR, "uploads")
legacy_generated_dir = os.path.join(config.BACKEND_DIR, "generated")
os.makedirs(legacy_upload_dir, exist_ok=True)
os.makedirs(legacy_generated_dir, exist_ok=True)

# Imports after config setup
from database import init_db, engine, Job
from sqlmodel import Session, select
from routes import router as api_router
from events import event_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Milimo Video Backend...")
    init_db()
    
    # Initialize Model Manager (Lazy load, but we can warm up if needed)
    # from model_engine import manager
    
    # Capture main loop for thread-safe operations in tasks
    import job_utils
    import asyncio
    job_utils.global_loop = asyncio.get_running_loop()
    
    # Startup Cleanup: Fail any jobs that were 'processing' when server died
    try:
        with Session(engine) as session:
            zombie_jobs = session.exec(select(Job).where(Job.status == "processing")).all()
            if zombie_jobs:
                logger.warning(f"Found {len(zombie_jobs)} zombie jobs from previous run. Marking as failed.")
                for job in zombie_jobs:
                    job.status = "failed"
                    job.error_message = "Server Restarted (Zombie Job)"
                    session.add(job)
                session.commit()
    except Exception as e:
        logger.error(f"Startup cleanup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Signal threads to stop
    count = job_utils.cancel_all_jobs()
    if count > 0:
        logger.info(f"Waiting for {count} jobs to cancel...")
        # Give threads a chance to hit the cancellation check
        await asyncio.sleep(2.0)

app = FastAPI(lifespan=lifespan)

# CORS - Belt and suspenders approach:
# 1. CORSMiddleware handles preflight OPTIONS and standard API responses
# 2. Raw HTTP middleware guarantees CORS headers on ALL responses (incl. errors, static files)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """Guarantee CORS headers on every response, including errors."""
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                "Access-Control-Allow-Headers": "Range, Content-Type, Authorization",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
                "Access-Control-Max-Age": "86400",
            },
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
    return response

# Include Routers (Must be before static mounts to avoid shadowing)
app.include_router(api_router)

# --- Safari-Compatible Media Serving (Range Requests) ---
# Safari REQUIRES HTTP Range requests (206 Partial Content) for <video>/<audio> playback.
# FastAPI's StaticFiles doesn't always handle Range headers properly for media.
# This endpoint intercepts media file requests and serves them with full Range support.
MEDIA_EXTENSIONS = {'.mp4', '.webm', '.mp3', '.wav', '.ogg', '.m4a', '.aac'}
MEDIA_DIRS = {
    "/projects": config.PROJECTS_DIR,
    "/uploads": legacy_upload_dir,
    "/generated": legacy_generated_dir,
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS, HEAD",
    "Access-Control-Allow-Headers": "Range, Content-Type",
    "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
}

@app.get("/projects/{file_path:path}")
@app.get("/uploads/{file_path:path}")
@app.get("/generated/{file_path:path}")
async def serve_media(request: Request, file_path: str):
    """Serve media files with HTTP Range request support for Safari."""
    # Determine which base directory to use from the route
    route_prefix = request.url.path.split('/')[1]  # 'projects', 'uploads', or 'generated'
    base_dir = MEDIA_DIRS.get(f"/{route_prefix}")
    if not base_dir:
        return Response(status_code=404)

    full_path = os.path.join(base_dir, file_path)
    full_path = os.path.normpath(full_path)

    # Security: Prevent path traversal
    if not full_path.startswith(os.path.normpath(base_dir)):
        return Response(status_code=403)

    if not os.path.isfile(full_path):
        return Response(status_code=404)

    # Determine content type
    content_type, _ = mimetypes.guess_type(full_path)
    if not content_type:
        content_type = "application/octet-stream"

    file_size = os.path.getsize(full_path)
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range header: "bytes=start-end"
        range_spec = range_header.strip().lower()
        if range_spec.startswith("bytes="):
            range_spec = range_spec[6:]
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1

            # Clamp to valid range
            start = max(0, start)
            end = min(end, file_size - 1)
            content_length = end - start + 1

            with open(full_path, "rb") as f:
                f.seek(start)
                data = f.read(content_length)

            return Response(
                content=data,
                status_code=206,
                headers={
                    **CORS_HEADERS,
                    "Content-Type": content_type,
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Cache-Control": "public, max-age=3600",
                },
            )

    # No Range header — serve full file with Accept-Ranges hint
    with open(full_path, "rb") as f:
        data = f.read()

    return Response(
        content=data,
        status_code=200,
        headers={
            **CORS_HEADERS,
            "Content-Type": content_type,
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600",
        },
    )

# Static Mounts - Fallback for non-media files (HTML, images, etc.)
# Note: The media endpoint above takes priority for matching paths due to route precedence.
app.mount("/projects", StaticFiles(directory=config.PROJECTS_DIR), name="projects")
app.mount("/uploads", StaticFiles(directory=legacy_upload_dir), name="uploads")
app.mount("/generated", StaticFiles(directory=legacy_generated_dir), name="generated") # Legacy

# Event Manager Endpoint
@app.get("/events")
async def events(request: Request):
    return await event_manager.subscribe(request)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=config.API_PORT, reload=True)
