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
    # Storyboard
    scene_id: Optional[str] = None
    action: Optional[str] = None
    dialogue: Optional[str] = None
    character: Optional[str] = None
    shot_type: Optional[str] = None
    timeline: List[TimelineItem] = []

class GenerateAdvancedRequest(BaseModel):
    project_id: str
    shot_config: ShotConfig

class ProjectState(BaseModel):
    id: str
    name: str
    shots: List[dict]
    scenes: List[dict] = []
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42
    created_at: Optional[float] = None
    updated_at: Optional[float] = None

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

class ElementUpdate(BaseModel):
    """Partial update for an element — all fields optional."""
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    trigger_word: Optional[str] = None
    image_path: Optional[str] = None

class ElementVisualizeRequest(BaseModel):
    prompt_override: Optional[str] = None
    guidance_scale: float = 2.0
    enable_ae: bool = True

class InpaintRequest(BaseModel):
    image_path: str
    mask_path: Optional[str] = None
    points: Optional[str] = None  # For SAM point-based
    text_mask: Optional[str] = None  # For SAM text-based segmentation
    prompt: str

class DetectRequest(BaseModel):
    image_path: str
    text: str
    confidence: float = 0.5

class SegmentTextRequest(BaseModel):
    image_path: str
    text: str
    confidence: float = 0.5

class ScriptParseRequest(BaseModel):
    script_text: str
    parse_mode: Optional[str] = "auto"  # auto, screenplay, freeform, numbered

# ── Typed Storyboard Schemas ──────────────────────────────────────────

class StoryboardShotData(BaseModel):
    """Typed shot data for storyboard commit (replaces dict)."""
    action: Optional[str] = None
    dialogue: Optional[str] = None
    character: Optional[str] = None
    shot_type: Optional[str] = None

class StoryboardSceneData(BaseModel):
    """Typed scene data for storyboard commit (replaces dict)."""
    name: str = "Scene 1"
    content: Optional[str] = None
    shots: List[StoryboardShotData] = []

class CommitStoryboardRequest(BaseModel):
    scenes: List[StoryboardSceneData]

class ReorderShotsRequest(BaseModel):
    """Ordered list of shot IDs — index in list becomes new Shot.index."""
    scene_id: str
    shot_ids: List[str]

class AddShotRequest(BaseModel):
    """Add a manual shot to a scene."""
    scene_id: str
    action: Optional[str] = "A cinematic shot..."
    dialogue: Optional[str] = None
    character: Optional[str] = None
    shot_type: Optional[str] = "medium"

class BatchGenerateRequest(BaseModel):
    """Generate multiple shots sequentially."""
    shot_ids: List[str]

class UpdateSceneRequest(BaseModel):
    """Update a scene's properties."""
    name: Optional[str] = None

class ReorderScenesRequest(BaseModel):
    """Ordered list of scene IDs — index in list becomes new Scene.index."""
    scene_ids: List[str]

class ProjectSettingsUpdate(BaseModel):
    """Partial update for project settings — all fields optional."""
    name: Optional[str] = None
    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = None

class BatchThumbnailRequest(BaseModel):
    """Generate concept art thumbnails for multiple shots."""
    shot_ids: List[str]
    width: Optional[int] = 512
    height: Optional[int] = 320
    force: bool = False  # Re-generate even if thumbnail exists

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

class TrackingSaveRequest(BaseModel):
    session_id: str
    video_path: Optional[str] = None
    frames: List[dict]  # List of {frame_idx, num_objects, masks: {obj_id: b64}, scores: {obj_id: float}}
