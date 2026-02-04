import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(config.PROJECTS_DIR, exist_ok=True)
legacy_upload_dir = os.path.join(config.BACKEND_DIR, "uploads")
legacy_generated_dir = os.path.join(config.BACKEND_DIR, "generated")
os.makedirs(legacy_upload_dir, exist_ok=True)
os.makedirs(legacy_generated_dir, exist_ok=True)

# Imports after config setup
from database import init_db
from routes import router as api_router
from events import event_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Milimo Video Backend...")
    init_db()
    
    # Initialize Model Manager (Lazy load, but we can warm up if needed)
    # from model_engine import manager
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers (Must be before static mounts to avoid shadowing)
app.include_router(api_router)

# Static Mounts - CRITICAL: Preserve these for external compatibility
app.mount("/projects", StaticFiles(directory=config.PROJECTS_DIR), name="projects")
app.mount("/uploads", StaticFiles(directory=legacy_upload_dir), name="uploads")
app.mount("/generated", StaticFiles(directory=legacy_generated_dir), name="generated") # Legacy

# Event Manager Endpoint
@app.get("/events")
async def events(request: Request):
    return await event_manager.subscribe(request)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=config.API_PORT, reload=True)
