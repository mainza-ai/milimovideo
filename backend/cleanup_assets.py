import os
import sys
from sqlmodel import Session, select
from database import engine, Asset, Job
import config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanup")

def cleanup():
    """
    Scans Assets and Jobs. If the file on disk is missing, removes the DB record.
    """
    config.setup_paths()
    
    with Session(engine) as session:
        # 1. Cleanup Assets
        logger.info("Scanning Assets...")
        assets = session.exec(select(Asset)).all()
        assets_removed = 0
        
        for asset in assets:
            # Resolve Path
            full_path = asset.path
            
            # If path is relative or web-url (legacy data might have this), try to resolve
            if not os.path.isabs(full_path):
                 if full_path.startswith("/projects/"):
                     rel = full_path[len("/projects/"):]
                     full_path = os.path.join(config.PROJECTS_DIR, rel)
            
            if not os.path.exists(full_path):
                logger.warning(f"Asset missing on disk: {asset.filename} (ID: {asset.id}) -> Removing DB Record.")
                session.delete(asset)
                assets_removed += 1
            else:
                # logger.info(f"Verified: {asset.filename}")
                pass
                
        # 2. Cleanup Jobs (Completed)
        logger.info("Scanning Completed Jobs...")
        jobs = session.exec(select(Job).where(Job.status == "completed")).all()
        jobs_updated = 0
        
        for job in jobs:
             if not job.output_path:
                 continue
                 
             full_path = job.output_path
             if not os.path.isabs(full_path):
                 if full_path.startswith("/projects/"):
                     rel = full_path[len("/projects/"):]
                     full_path = os.path.join(config.PROJECTS_DIR, rel)
            
             if not os.path.exists(full_path):
                 logger.warning(f"Job Output missing on disk: {job.id} -> Marking as Deleted.")
                 job.status = "deleted"
                 job.output_path = None
                 session.add(job)
                 jobs_updated += 1

        session.commit()
        logger.info(f"Cleanup Complete. Removed {assets_removed} Assets. Updated {jobs_updated} Jobs.")

if __name__ == "__main__":
    cleanup()
