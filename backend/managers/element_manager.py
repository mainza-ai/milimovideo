import logging
from typing import List, Optional
from sqlmodel import Session, select
from sqlmodel import Session, select
from database import engine, Element, Project, Asset
from datetime import datetime, timezone
import uuid

logger = logging.getLogger("element_manager")

class ElementManager:
    def __init__(self):
        pass

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

    def inject_elements_into_prompt(self, prompt: str, project_id: str) -> str:
        """
        Scans values in the prompt for trigger words (e.g. @Hero) and 
        replaces them with the element's description.
        """
        # Get all elements for this project
        elements = self.get_elements(project_id)
        if not elements:
            return prompt

        final_prompt = prompt
        injected_count = 0
        
        # Simple replacement for now. 
        # Future: Use LLM for smarter insertion if needed.
        for el in elements:
            if el.trigger_word and el.trigger_word in final_prompt:
                # Replace "@Hero" with "A tall woman... (Hero)"
                # We keep the name in brackets for clarity or just replace it?
                # Let's replace: "A tall woman... "
                
                # Check if we should append name to description to clarify identity?
                # description usually has physical traits.
                replacement = f"{el.description}" 
                final_prompt = final_prompt.replace(el.trigger_word, replacement)
                injected_count += 1
        
        if injected_count > 0:
            logger.info(f"Injected {injected_count} elements into prompt for Project {project_id}")
            
        return final_prompt

    async def generate_visual(self, element_id: str, prompt_override: str = None) -> Optional[str]:
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
            
            # Enhance prompt for "Character Sheet" style if it's a character
            if element.type.lower() == 'character':
                final_prompt = f"Character Sheet Design for {element.name}: {base_prompt}. Full body character turnaround, white background, high quality, concept art style."
            elif element.type.lower() == 'location':
                final_prompt = f"Concept art of {element.name}: {base_prompt}. Wide shot, atmospheric, cinematic lighting, high quality."
            else:
                final_prompt = f"Concept design of {element.name}: {base_prompt}. Product studio lighting, neutral background."
            
            logger.info(f"Generating visual for Element {element.name} ({element.id})")
            
            # Run Flux T2I (Offload to thread if needed, but manager is async-aware? 
            # Flux wrapper calls are blocking on GPU. In production we'd use a queue.
            # For MVP Agentic Mode, we run it directly (blocking the worker thread but it's okay for single user).
            
            try:
                image = flux_inpainter.generate_image(
                    prompt=final_prompt,
                    width=1024,
                    height=1024,
                    guidance=3.5 
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
                return None

# Singleton
element_manager = ElementManager()
