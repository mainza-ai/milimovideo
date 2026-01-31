#!/usr/bin/env python3
"""
Migration: Create _global project and migrate orphaned assets to project workspaces
"""
import os
import sys
import shutil
import logging
from pathlib import Path

# Add parent directory to path to import database module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import engine, Project, Asset, Job
from sqlmodel import Session, select
from datetime import datetime, timezone
import uuid

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECTS_DIR = os.path.join(BACKEND_DIR, "projects")
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
GENERATED_DIR = os.path.join(BACKEND_DIR, "generated")

def create_global_project():
    """Create _global project for orphaned assets"""
    logger.info("Creating _global project workspace...")
    
    with Session(engine) as session:
        # Check if _global project exists
        global_project = session.get(Project, "_global")
        
        if not global_project:
            global_project = Project(
                id="_global",
                name="Global Assets (Legacy)",
                shots=[],
                resolution_w=768,
                resolution_h=512,
                fps=25,
                seed=42,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(global_project)
            session.commit()
            logger.info("✅ Created _global project in database")
        else:
            logger.info("ℹ️  _global project already exists")
    
    # Create workspace directories
    global_workspace = os.path.join(PROJECTS_DIR, "_global")
    for subdir in ["assets", "generated", "thumbnails", "workspace"]:
        path = os.path.join(global_workspace, subdir)
        os.makedirs(path, exist_ok=True)
    
    logger.info(f"✅ Created _global workspace at {global_workspace}")

def migrate_orphaned_uploads():
    """Move orphaned uploads to _global project"""
    logger.info("Migrating orphaned uploads...")
    
    if not os.path.exists(UPLOAD_DIR):
        logger.info("ℹ️  No uploads directory found")
        return
    
    upload_files = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith(".") and "_thumb" not in f]
    
    if not upload_files:
        logger.info("ℹ️  No orphaned upload files found")
        return
    
    global_assets_dir = os.path.join(PROJECTS_DIR, "_global", "assets")
    moved_count = 0
    
    with Session(engine) as session:
        for filename in upload_files:
            old_path = os.path.join(UPLOAD_DIR, filename)
            new_path = os.path.join(global_assets_dir, filename)
            
            # Check if asset exists in DB
            asset = session.exec(select(Asset).where(Asset.filename == filename)).first()
            
            if asset:
                # Update existing asset to point to new location
                asset.project_id = "_global"
                asset.path = f"_global/assets/{filename}"
                asset.url = f"/projects/_global/assets/{filename}"
                logger.info(f"  Updated asset {asset.id}: {filename}")
            else:
                # Create new asset record for orphan
                ext = filename.split(".")[-1].lower()
                asset = Asset(
                    id=uuid.uuid4().hex,
                    project_id="_global",
                    type="video" if ext in ["mp4", "mov"] else "image",
                    path=f"_global/assets/{filename}",
                    url=f"/projects/_global/assets/{filename}",
                    filename=filename,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(asset)
                logger.info(f"  Created asset {asset.id}: {filename}")
            
            # Move physical file
            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
                moved_count += 1
        
        session.commit()
    
    logger.info(f"✅ Migrated {moved_count} orphaned uploads to _global/assets/")

def migrate_orphaned_generated():
    """Move orphaned generated files to _global project"""
    logger.info("Migrating orphaned generated files...")
    
    if not os.path.exists(GENERATED_DIR):
        logger.info("ℹ️  No generated directory found")
        return
    
    gen_files = [
        f for f in os.listdir(GENERATED_DIR) 
        if not f.startswith(".") and "_thumb" not in f and "_list" not in f 
        and "_last" not in f and "_part" not in f
    ]
    
    if not gen_files:
        logger.info("ℹ️  No orphaned generated files found")
        return
    
    global_generated_dir = os.path.join(PROJECTS_DIR, "_global", "generated")
    global_thumbnails_dir = os.path.join(PROJECTS_DIR, "_global", "thumbnails")
    moved_count = 0
    
    with Session(engine) as session:
        for filename in gen_files:
            job_id = filename.split(".")[0]
            old_path = os.path.join(GENERATED_DIR, filename)
            new_path = os.path.join(global_generated_dir, filename)
            
            # Check if job exists
            job = session.get(Job, job_id)
            
            if job:
                # Update job to point to new location
                job.project_id = job.project_id or "_global"
                job.output_path = f"/projects/{job.project_id}/generated/{filename}"
                
                # Handle thumbnail
                if job.thumbnail_path:
                    old_thumb_name = os.path.basename(job.thumbnail_path)
                    old_thumb_path = os.path.join(GENERATED_DIR, old_thumb_name)
                    new_thumb_path = os.path.join(global_thumbnails_dir, old_thumb_name)
                    
                    if os.path.exists(old_thumb_path):
                        shutil.move(old_thumb_path, new_thumb_path)
                    
                    job.thumbnail_path = f"/projects/{job.project_id}/thumbnails/{old_thumb_name}"
                
                logger.info(f"  Updated job {job_id}")
            else:
                # Create legacy job record
                job = Job(
                    id=job_id,
                    project_id="_global",
                    type="legacy",
                    status="completed",
                    created_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    output_path=f"/projects/_global/generated/{filename}",
                    prompt="Legacy File (Migrated)"
                )
                session.add(job)
                logger.info(f"  Created legacy job {job_id}")
            
            # Move physical file
            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
                moved_count += 1
        
        session.commit()
    
    logger.info(f"✅ Migrated {moved_count} orphaned generated files to _global/generated/")

def migrate_project_jobs():
    """Update existing project jobs to use project-scoped paths"""
    logger.info("Updating existing project jobs...")
    
    with Session(engine) as session:
        # Get all jobs with output paths
        jobs = session.exec(select(Job).where(Job.output_path != None)).all()
        updated_count = 0
        
        for job in jobs:
            # Skip if already migrated (starts with /projects/)
            if job.output_path and job.output_path.startswith("/projects/"):
                continue
            
            # Extract filename from old path
            if job.output_path:
                filename = os.path.basename(job.output_path)
                project_id = job.project_id or "_global"
                
                # Update to project-scoped path
                job.output_path = f"/projects/{project_id}/generated/{filename}"
                updated_count += 1
            
            # Update thumbnail path
            if job.thumbnail_path and not job.thumbnail_path.startswith("/projects/"):
                thumb_filename = os.path.basename(job.thumbnail_path)
                project_id = job.project_id or "_global"
                job.thumbnail_path = f"/projects/{project_id}/thumbnails/{thumb_filename}"
        
        session.commit()
        logger.info(f"✅ Updated {updated_count} job paths to project-scoped format")

def verify_migration():
    """Verify migration completed successfully"""
    logger.info("\n" + "="*50)
    logger.info("MIGRATION VERIFICATION")
    logger.info("="*50)
    
    with Session(engine) as session:
        # Check _global project
        global_proj = session.get(Project, "_global")
        logger.info(f"✅ _global project exists: {global_proj is not None}")
        
        # Check assets
        assets = session.exec(select(Asset)).all()
        orphan_assets = [a for a in assets if a.project_id is None]
        logger.info(f"📊 Total assets: {len(assets)}")
        logger.info(f"📊 Orphaned assets (project_id=None): {len(orphan_assets)}")
        
        # Check jobs
        jobs = session.exec(select(Job)).all()
        legacy_paths = [j for j in jobs if j.output_path and not j.output_path.startswith("/projects/")]
        logger.info(f"📊 Total jobs: {len(jobs)}")
        logger.info(f"📊 Jobs with legacy paths: {len(legacy_paths)}")
    
    # Check filesystem
    global_workspace = os.path.join(PROJECTS_DIR, "_global")
    if os.path.exists(global_workspace):
        assets_count = len([f for f in os.listdir(os.path.join(global_workspace, "assets")) if not f.startswith(".")])
        generated_count = len([f for f in os.listdir(os.path.join(global_workspace, "generated")) if not f.startswith(".")])
        logger.info(f"📁 _global/assets files: {assets_count}")
        logger.info(f"📁 _global/generated files: {generated_count}")
    
    logger.info("="*50 + "\n")

def main():
    logger.info("🚀 Starting Project Workspace Migration\n")
    
    try:
        # Step 1: Create _global project
        create_global_project()
        
        # Step 2: Migrate orphaned uploads
        migrate_orphaned_uploads()
        
        # Step 3: Migrate orphaned generated files
        migrate_orphaned_generated()
        
        # Step 4: Update existing project jobs
        migrate_project_jobs()
        
        # Step 5: Verify migration
        verify_migration()
        
        logger.info("✅ Migration completed successfully!")
        logger.info("\n💡 Next steps:")
        logger.info("  1. Restart the backend server")
        logger.info("  2. Test project creation and asset upload")
        logger.info("  3. Verify frontend can load projects correctly")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
