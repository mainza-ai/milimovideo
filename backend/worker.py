import logging
import config

# Setup paths first
config.setup_paths()

# Re-export Model Manager
from model_engine import manager, ModelManager

# Re-export Tasks (for backward compatibility if anything imports them)
from tasks.video import generate_video_task, generate_standard_video_task
from tasks.chained import generate_chained_video_task
from tasks.image import generate_image_task

# Re-export Utils
from job_utils import update_job_db, active_jobs, update_shot_db, broadcast_progress
from file_utils import get_base_dir, get_project_output_paths

logger = logging.getLogger(__name__)

# If this file is run directly, it might be intended to run a standalone worker process
# (though currently Milimo uses FastAPI BackgroundTasks in server process).
# We keep this for compatibility or future separation.

if __name__ == "__main__":
    logger.info("Worker module loaded. This module provides task definitions for the server.")
