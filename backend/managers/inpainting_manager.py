import logging
import requests
import io
import os
from PIL import Image
import config
from models.flux_wrapper import flux_inpainter
from database import Job, get_session
import uuid

logger = logging.getLogger("inpainting_manager")

class InpaintingManager:
    def __init__(self):
        self.sam_url = f"http://localhost:{config.SAM_SERVICE_PORT}"

    async def get_mask_from_sam(self, image_path: str, points: list) -> str:
        """
        Calls SAM microservice to get a mask.
        Returns path to saved mask.
        """
        if not os.path.exists(image_path):
             logger.error(f"Image not found: {image_path}")
             return None

        # Prepare file
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}  # Field name matches server: image
                # points should be list of lists
                data = {"points": str(points), "multimask": "false"} 
                
                logger.info(f"Requesting mask from SAM at {self.sam_url} with points {points}")
                res = requests.post(f"{self.sam_url}/predict/mask", files=files, data=data)
                
                if res.status_code == 200:
                    # Response is a PNG image
                    mask_content = res.content
                    
                    # Save mask to same dir as image but with suffix
                    dir_name = os.path.dirname(image_path)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    mask_filename = f"{base_name}_mask_{uuid.uuid4().hex[:6]}.png"
                    mask_path = os.path.join(dir_name, mask_filename)
                    
                    with open(mask_path, "wb") as f_out:
                         f_out.write(mask_content)
                         
                    logger.info(f"Mask saved to {mask_path}")
                    return mask_path
                else:
                    logger.error(f"SAM Service failed: {res.text}")
                    return None
        except Exception as e:
            logger.error(f"SAM Service connection failed: {e}")
            return None
        return None

    async def process_inpaint(self, job_id: str, image_path: str, mask_path: str, prompt: str):
        """
        Runs Flux In-Painting on the image using the mask.
        """
        try:
            logger.info(f"Starting Inpaint Job {job_id} with prompt: {prompt}")
            
            if not os.path.exists(image_path):
                 raise FileNotFoundError(f"Source image not found: {image_path}")
            if not os.path.exists(mask_path):
                 raise FileNotFoundError(f"Mask not found: {mask_path}")

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Run Inference
            # This is synchronous and blocking, but running in BackgroundTasks so acceptable for MVP
            result = flux_inpainter.inpaint(image, mask, prompt, guidance=2.0, enable_ae=True, enable_true_cfg=False)
            
            # Save result in projects folder if possible, else temp
            # Try to derive project path from image path if it follows convention
            # /projects/{id}/assets/... -> /projects/{id}/generated/...
            
            out_dir = os.path.dirname(image_path)
            # If in assets, move to generated? Or keep with assets?
            # Standard output location:
            # projects/{id}/generated/
            
            if "assets" in out_dir:
                 out_dir = out_dir.replace("assets", "generated")
            
            os.makedirs(out_dir, exist_ok=True)
            
            out_filename = f"inpaint_{job_id}.jpg"
            out_path = os.path.join(out_dir, out_filename)
            
            result.save(out_path)
            logger.info(f"In-painting completed: {out_path}")
            
            # Update Job status in DB?
            # The current server logic doesn't create a full 'Job' record for edits yet, 
            # just fires this off. But usually we want to track it.
            # Ideally the caller (server.py) creates a pending Job, and we update it here.
            # But the 'job_id' passed in is just a string.
            # Let's try to update if it exists in DB.
            
            # TODO: Add Job update logic here
            
            return out_path
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise e

inpainting_manager = InpaintingManager()
