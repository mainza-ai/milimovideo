import logging
from typing import List, Optional
from sqlmodel import Session, select
from sqlmodel import Session, select
from database import engine, Element, Project, Asset
from datetime import datetime, timezone
import uuid
import os

logger = logging.getLogger("element_manager")

class ElementManager:
    def __init__(self):
        pass

    async def generate_visual_task(self, job_id: str, element_id: str, prompt_override: str = None, guidance: float = 2.0, enable_ae: bool = False):
        """Background task wrapper for generate_visual with job tracking."""
        from job_utils import active_jobs, update_job_progress, update_job_db
        from database import Job, engine
        from sqlmodel import Session
        from datetime import datetime, timezone, timedelta
        import asyncio

        logger.info(f"Starting Element Visual Job {job_id} for Element {element_id}")
        
        # 1. Look up Project ID for tracking
        with Session(engine) as session:
            element = session.get(Element, element_id)
            if element:
                active_jobs[job_id]["project_id"] = element.project_id

        # 2. Update Status
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["status_message"] = "Generating..."
        update_job_progress(job_id, 10, "Initializing...")

        try:
            # 3. generate_visual is async but internally blocking (Flux). 
            # We need to modify generate_visual to accept cancellation check callback or job_id?
            # Or just check before calling core generation.
            
            # Since generate_visual was designed to return path, let's just call it.
            # BUT we want to support 'enable_ae'.
            # We need to update existing generate_visual signature first.
            path = await self.generate_visual(element_id, prompt_override, guidance, enable_ae, job_id)
            
            if path:
                # Success
                active_jobs[job_id]["status"] = "completed"
                active_jobs[job_id]["progress"] = 100
                active_jobs[job_id]["status_message"] = "Completed"
                # Need to update DB job record too if we want persistence, 
                # but currently we don't create a DB Job record for elements in the route!
                # We only created in active_jobs.
                # If we want detailed history, request said "active job recovery".
                # For now active_jobs in memory is enough for session persistence.
            else:
                # Failed (or cancelled inside)
                if active_jobs[job_id].get("cancelled"):
                    active_jobs[job_id]["status"] = "cancelled"
                else:
                    active_jobs[job_id]["status"] = "failed"
                    active_jobs[job_id]["status_message"] = "Generation failed"

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["status_message"] = str(e)


    def create_element(self, project_id: str, name: str, type: str, description: str, trigger_word: str = None, image_path: str = None) -> Element:
        """Create a new story element."""
        if not trigger_word:
            trigger_word = f"@{name.replace(' ', '')}"
            
        with Session(engine) as session:
            element = Element(
                project_id=project_id,
                name=name,
                type=type,
                description=description,
                trigger_word=trigger_word,
                image_path=image_path
            )
            session.add(element)
            session.commit()
            session.refresh(element)
            return element

    def get_elements(self, project_id: str) -> List[Element]:
        """List all elements for a project."""
        with Session(engine) as session:
            return session.exec(select(Element).where(Element.project_id == project_id)).all()

    def update_element(self, element_id: str, updates: dict) -> Optional[Element]:
        """Update an element."""
        with Session(engine) as session:
            element = session.get(Element, element_id)
            if not element:
                return None
            
            for k, v in updates.items():
                if hasattr(element, k):
                    setattr(element, k, v)
            
            session.add(element)
            session.commit()
            session.refresh(element)
            return element

    def delete_element(self, element_id: str) -> bool:
        """Delete an element."""
        with Session(engine) as session:
            element = session.get(Element, element_id)
            if not element:
                return False
            session.delete(element)
            session.commit()
            return True


    
    def inject_elements_into_prompt(self, prompt: str, project_id: str) -> tuple[str, List[str]]:
        """
        Scans prompt for trigger words.
        Returns: (enhanced_prompt, list_of_image_paths)
        """
        # Get all elements for this project
        elements = self.get_elements(project_id)
        if not elements:
            return prompt, []

        final_prompt = prompt
        injected_count = 0
        collected_images = []
        
        for el in elements:
            if el.trigger_word and el.trigger_word in final_prompt:
                # If element has a visual, use Name (avoid description contamination)
                # If text-only, use Description
                if el.image_path:
                    replacement = el.name
                else:
                    replacement = f"{el.description}"
                    
                final_prompt = final_prompt.replace(el.trigger_word, replacement)
                injected_count += 1
                
                # Collect Visual Asset if available
                if el.image_path:
                    # Resolve full path if it's a relative web URL
                    # stored as /projects/{id}/assets/...
                    if el.image_path.startswith("/projects"):
                        from config import PROJECTS_DIR
                        # Remove /projects/ prefix to join with PROJECTS_DIR parent? 
                        # Actually PROJECTS_DIR is /.../projects
                        # URL: /projects/123/assets/img.jpg
                        # File: PROJECTS_DIR/123/assets/img.jpg
                        
                        relative = el.image_path.lstrip("/projects/") # -> 123/assets/img.jpg
                        full_path = os.path.join(PROJECTS_DIR, relative)
                        if os.path.exists(full_path):
                            collected_images.append(full_path)
                    elif os.path.exists(el.image_path):
                        collected_images.append(el.image_path)
        
        if injected_count > 0:
            logger.info(f"Injected {injected_count} elements into prompt. Found {len(collected_images)} visuals.")
            
        return final_prompt, collected_images

    async def generate_visual(self, element_id: str, prompt_override: str = None, guidance: float = 2.0, enable_ae: bool = False, job_id: str = None) -> Optional[str]:
        """
        Uses Flux to generate a visual representation for the element.
        Updates the element's image_path and returns the url/path.
        """
        from models.flux_wrapper import flux_inpainter
        import os
        from config import PROJECTS_DIR
        
        with Session(engine) as session:
            element = session.get(Element, element_id)
            if not element:
                logger.error(f"Element {element_id} not found")
                return None
            
            project_id = element.project_id
            
            # Construct Prompt
            # If override provided, use it. Otherwise construct from description.
            # "Character Design Sheet for [Name]: [Description]. Full body, white background..."
            
            base_prompt = prompt_override if prompt_override else element.description
        
        # Inject other elements if triggers exist in the prompt
        if "@" in base_prompt:
             # We need to import self to call instance method? No, self is available.
             # But this method is async.
             # inject_elements_into_prompt is synchronous.
             base_prompt, element_images = self.inject_elements_into_prompt(base_prompt, project_id)
             # NOTE: We currently ignore element_images for element generation (Flux T2I)
             # unless we update generate_image to take them.
             # START_FIX: Pass these images to generate_image below if found!
        else:
             element_images = []

        # Enhance prompt for "Character Sheet" style if it's a character
        if element.type.lower() == 'character':
            final_prompt = f"Character Sheet Design for {element.name}: {base_prompt}. Full body character turnaround, white background, high quality, concept art style."
        elif element.type.lower() == 'location':
            final_prompt = f"Concept art of {element.name}: {base_prompt}. Wide shot, atmospheric, cinematic lighting, high quality."
        else:
            final_prompt = f"Concept design of {element.name}: {base_prompt}. Product studio lighting, neutral background."
            
        logger.info(f"Generating visual for Element {element.name} ({element.id})")
        
        # Check Cancellation
        if job_id:
            from job_utils import active_jobs
            if job_id in active_jobs and active_jobs[job_id].get("cancelled", False):
                logger.info(f"Job {job_id} cancelled before generation")
                return None

        # Run Flux T2I (Offload to thread if needed, but manager is async-aware? 
        # Flux wrapper calls are blocking on GPU. In production we'd use a queue.
        # For MVP Agentic Mode, we run it directly (blocking the worker thread but it's okay for single user).
        
        try:
            image = flux_inpainter.generate_image(
                prompt=final_prompt,
                width=1024,
                height=1024,
                guidance=guidance,
                enable_true_cfg=False, # Disable True CFG for element generation to ensure stability
                enable_ae=enable_ae, # User controlled (default True)
                ip_adapter_images=element_images # Pass visual conditioning found from triggers!
            )
                
            # Save Image
            # New Path: /projects/{id}/assets/elements/visual_{element_id}_{suffix}.jpg
            
            assets_dir = os.path.join(PROJECTS_DIR, project_id, "assets", "elements")
            os.makedirs(assets_dir, exist_ok=True)
            
            # Use random suffix to avoid caching issues on regeneration
            suffix = uuid.uuid4().hex[:6]
            filename = f"visual_{element.id}_{suffix}.jpg"
            save_path = os.path.join(assets_dir, filename)
            
            image.save(save_path, quality=95)
            logger.info(f"Saved element visual to {save_path}")
            
            # Create Asset Record
            web_url = f"/projects/{project_id}/assets/elements/{filename}"
            
            asset_id = uuid.uuid4().hex
            new_asset = Asset(
                id=asset_id,
                project_id=project_id,
                type="image",
                path=save_path,
                url=web_url,
                filename=filename,
                created_at=datetime.now(timezone.utc),
                width=image.width,
                height=image.height
            )
            session.add(new_asset)
            
            # Update Element with the RELATIVE asset path (or just use the web_url)
            # We'll use the relative path from project root for portability if needed, 
            # but for simplicity let's store the web_url or the relative path from assets.
            # Let's store the web_url for easy frontend usage.
            element.image_path = web_url
            
            session.add(element)
            session.commit()
            session.refresh(element)
            
            return save_path
            
        except Exception as e:
            logger.error(f"Visual generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Singleton
element_manager = ElementManager()
