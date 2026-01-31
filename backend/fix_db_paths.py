
import os
import sys
from sqlmodel import Session, select
from database import engine, Job, Project, Asset

# Add current dir to path to import database/models
sys.path.append(os.path.dirname(__file__))

def fix_paths():
    print("Starting DB Path Fixer...")
    
    with Session(engine) as session:
        jobs = session.exec(select(Job)).all()
        print(f"Found {len(jobs)} jobs.")
        
        fixed_count = 0
        
        for job in jobs:
            if not job.project_id:
                continue
                
            changed = False
            
            # 1. Fix Output Path logic
            # Expected: /projects/{id}/generated/{filename}
            # Current might be: /generated/{filename}
            if job.output_path:
                filename = os.path.basename(job.output_path)
                expected_output_url = f"/projects/{job.project_id}/generated/{filename}"
                
                # Check if file exists at expected location
                phys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "projects", job.project_id, "generated", filename))
                
                if os.path.exists(phys_path):
                    if job.output_path != expected_output_url:
                        print(f"Fixing Output Path for {job.id}: {job.output_path} -> {expected_output_url}")
                        job.output_path = expected_output_url
                        changed = True
            
            # 2. Fix Thumbnail Path logic
            # Expected: /projects/{id}/thumbnails/{filename_thumb}
            # Current might be: /generated/{filename_thumb} or /projects/{id}/generated/...
            
            # Infer thumbnail filename
            thumb_filename = None
            if job.thumbnail_path:
                thumb_filename = os.path.basename(job.thumbnail_path)
            elif job.output_path:
                vid_name = os.path.basename(job.output_path)
                if vid_name.endswith(".mp4"):
                    thumb_filename = vid_name.replace(".mp4", "_thumb.jpg")
                else:
                    thumb_filename = vid_name # Image is its own thumbnail often? Or same name
            
            if thumb_filename:
                expected_thumb_url = f"/projects/{job.project_id}/thumbnails/{thumb_filename}"
                
                # Check if file exists in THUMBNAILS folder
                phys_thumb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "projects", job.project_id, "thumbnails", thumb_filename))
                
                if os.path.exists(phys_thumb_path):
                    if job.thumbnail_path != expected_thumb_url:
                        print(f"Fixing Thumbnail Path for {job.id}: {job.thumbnail_path} -> {expected_thumb_url}")
                        job.thumbnail_path = expected_thumb_url
                        changed = True
                else:
                    # Fallback: Check if it's in GENERATED folder (legacy separation)
                    phys_gen_thumb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "projects", job.project_id, "generated", thumb_filename))
                    if os.path.exists(phys_gen_thumb_path):
                        # Use generated URL
                        expected_gen_url = f"/projects/{job.project_id}/generated/{thumb_filename}"
                        if job.thumbnail_path != expected_gen_url:
                            print(f"Fixing Thumbnail Path (Legacy Generated) for {job.id}: {job.thumbnail_path} -> {expected_gen_url}")
                            job.thumbnail_path = expected_gen_url
                            changed = True
            
            if changed:
                session.add(job)
                fixed_count += 1
        
        # 3. Fix Assets (Uploads)
        assets = session.exec(select(Asset)).all()
        print(f"Found {len(assets)} assets.")
        asset_fixed = 0
        for asset in assets:
            changed = False
            # Check URL for legacy /uploads/
            if asset.url and asset.url.startswith("/uploads/"):
                # Should be project scoped? Or at least /projects/_uploads/
                # Check where file is
                filename = os.path.basename(asset.path)
                if asset.project_id:
                     expected = f"/projects/{asset.project_id}/assets/{filename}"
                     if asset.url != expected:
                         print(f"Fixing Asset URL {asset.id}: {asset.url} -> {expected}")
                         asset.url = expected
                         changed = True
                else:
                     # Legacy global
                     expected = f"/projects/_uploads/{filename}"
                     if asset.url != expected:
                         print(f"Fixing Asset URL {asset.id}: {asset.url} -> {expected}")
                         asset.url = expected
                         changed = True
            
            if changed:
                session.add(asset)
                asset_fixed += 1

        # 4. Fix Project Shots (JSON Blob)
        projects = session.exec(select(Project)).all()
        print(f"Found {len(projects)} projects.")
        proj_fixed = 0
        
        for project in projects:
            if not project.shots: continue
            shots_changed = False
            new_shots = []
            for shot in project.shots:
                # Check video_url and thumbnail_url
                if shot.get("video_url") and shot["video_url"].startswith("/generated/"):
                     # Fix it
                     filename = os.path.basename(shot["video_url"])
                     new_url = f"/projects/{project.id}/generated/{filename}"
                     print(f"Fixing Shot Video URL {shot.get('id')}: {shot['video_url']} -> {new_url}")
                     shot["video_url"] = new_url
                     shots_changed = True
                
                if shot.get("thumbnail_url") and shot["thumbnail_url"].startswith("http"):
                    # Check if path inside is legacy
                    if "/generated/" in shot["thumbnail_url"] and "/projects/" in shot["thumbnail_url"]:
                         # It is: .../projects/{id}/generated/... -> .../projects/{id}/thumbnails/...
                         # Try to fix if file exists in thumbnails
                         parts = shot["thumbnail_url"].split("/")
                         if "generated" in parts:
                             idx = parts.index("generated")
                             parts[idx] = "thumbnails"
                             new_url = "/".join(parts)
                             print(f"Fixing Shot Thumb URL {shot.get('id')}: generated -> thumbnails")
                             shot["thumbnail_url"] = new_url
                             shots_changed = True
                
                new_shots.append(shot)
            
            if shots_changed:
                project.shots = list(new_shots)
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(project, "shots")
                session.add(project)
                proj_fixed += 1

        session.commit()
        print(f"Fixed {fixed_count} jobs, {asset_fixed} assets, {proj_fixed} projects.")

if __name__ == "__main__":
    fix_paths()
