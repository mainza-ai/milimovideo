from pydantic import BaseModel
from typing import List, Optional, Literal

class TimelineItem(BaseModel):
    path: str
    frame_index: int = 0
    strength: float = 1.0
    type: Literal["image", "video"]

class ShotConfig(BaseModel):
    id: str
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: int = 42
    width: int = 768
    height: int = 512
    num_frames: int = 121
    fps: int = 25
    num_inference_steps: int = 40
    cfg_scale: float = 2.0
    enhance_prompt: bool = True
    upscale: bool = True
    pipeline_override: Optional[str] = "auto"
    auto_continue: bool = False
    timeline: List[TimelineItem] = []

class GenerateAdvancedRequest(BaseModel):
    project_id: str
    shot_config: ShotConfig

class ProjectState(BaseModel):
    id: str
    name: str
    shots: List[dict]
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42

class CreateProjectRequest(BaseModel):
    name: str
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42

class ElementCreate(BaseModel):
    name: str
    type: str
    description: str
    trigger_word: Optional[str] = None
    image_path: Optional[str] = None

class ElementVisualizeRequest(BaseModel):
    prompt_override: Optional[str] = None
    guidance_scale: float = 2.0
    enable_ae: bool = True

class InpaintRequest(BaseModel):
    image_path: str
    mask_path: Optional[str] = None
    points: Optional[str] = None # For SAM
    prompt: str

class ScriptParseRequest(BaseModel):
    script_text: str

class CommitStoryboardRequest(BaseModel):
    scenes: List[dict] # Should match parsed structure

class GenerateImageRequest(BaseModel):
    project_id: str
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 2.0
    seed: Optional[int] = None
    reference_images: List[str] = []
    enable_ae: bool = True
    enable_true_cfg: bool = False
