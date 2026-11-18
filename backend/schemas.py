from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional, Literal

class TimelineItem(BaseModel):
    path: str
    frame_index: int = 0
    strength: float = 1.0
    type: Literal["image", "video"]

    @classmethod
    def normalize(cls, value) -> list[dict]:
        """Accept canonical snake_case, legacy camelCase, or a JSON string."""
        import json as _json
        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = _json.loads(value) if value.strip() else []
            except _json.JSONDecodeError:
                return []
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized = {
                "path": item.get("path", ""),
                "frame_index": item.get("frame_index", item.get("frameIndex", 0)),
                "strength": item.get("strength", 1.0),
                "type": item.get("type", "image"),
            }
            try:
                result.append(cls(**normalized).model_dump())
            except Exception:
                continue
        return result

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
    priority: int = 50
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

class CaptionCue(BaseModel):
    id: Optional[str] = None
    start_frame: int = 0
    end_frame: int = 0
    text: str = ""


class TitleCue(BaseModel):
    """A burned-in title overlay positioned in normalized frame coordinates."""
    id: Optional[str] = None
    text: str = ""
    start_frame: int = 0
    end_frame: int = 0
    x: float = 0.5
    y: float = 0.5
    size: int = 48
    color: str = "#ffffff"
    box: bool = True


class SceneState(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    index: Optional[int] = None
    name: Optional[str] = None
    script_content: Optional[str] = None


class ShotState(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    project_id: Optional[str] = None
    scene_id: Optional[str] = None
    index: Optional[int] = None
    sequence_index: Optional[int] = None
    prompt: Optional[str] = ""
    width: int = 768
    height: int = 512
    seed: int = 42
    seed_locked: bool = True
    num_frames: int = 121
    num_inference_steps: int = 40
    priority: int = 50
    fps: Optional[int] = None
    cfg_scale: float = 2.0
    negative_prompt: Optional[str] = ""
    enhance_prompt: bool = True
    upscale: bool = False
    pipeline_override: Optional[str] = "auto"
    auto_continue: bool = False
    status: str = "pending"
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None
    track_index: int = 0
    start_frame: int = 0
    trim_in: int = 0
    trim_out: int = 0
    transition_in: int = 0
    action: Optional[str] = None
    dialogue: Optional[str] = None
    character: Optional[str] = None
    shot_type: Optional[str] = None
    timeline: List[Any] = []
    matched_elements: Optional[Any] = None
    keyframes: List[Any] = []
    blocking: List[Any] = []
    enhanced_prompt_result: Optional[str] = None
    last_job_id: Optional[str] = None
    selected_take_id: Optional[str] = None
    favorite_take_id: Optional[str] = None
    audio_gain: float = 1.0
    audio_muted: bool = False
    audio_fade_in: float = 0.0
    audio_fade_out: float = 0.0
    speed: float = 1.0
    created_at: Optional[Any] = None


class ProjectState(BaseModel):
    id: str
    name: str
    shots: List[ShotState]
    scenes: List[SceneState] = []
    captions: List[CaptionCue] = []
    captions_burn: bool = True
    titles: List[TitleCue] = []
    render_preset: Optional[str] = None
    render_crf: int = 18
    render_speed: str = "fast"
    render_audio_bitrate: str = "192k"
    render_audio_normalize: bool = False
    render_lut: Optional[str] = None
    script_content: Optional[str] = None
    resolution_w: int = 768
    resolution_h: int = 512
    fps: int = 25
    seed: int = 42
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    # None means "legacy client, skip the concurrency check". New clients send
    # the revision they read; a mismatch returns 409.
    revision: Optional[int] = None

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
    shot_id: Optional[str] = None

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
    matched_elements: Optional[List[dict]] = None  # [{element_id, element_name, element_type, ...}]

class StoryboardSceneData(BaseModel):
    """Typed scene data for storyboard commit (replaces dict)."""
    name: str = "Scene 1"
    content: Optional[str] = None
    shots: List[StoryboardShotData] = []

class CommitStoryboardRequest(BaseModel):
    scenes: List[StoryboardSceneData]
    script_text: Optional[str] = None

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


class ComfyWorkflowRequest(BaseModel):
    """A ComfyUI API-format workflow plus optional per-node input overrides."""
    workflow: Any
    overrides: Dict[str, Dict[str, Any]] = {}
    filename_prefix: Optional[str] = None
    live: bool = False


class ComfyUploadRequest(BaseModel):
    project_id: str
    url: str


class ComfySettingsRequest(BaseModel):
    url: Optional[str] = None
    enabled: Optional[bool] = None
    declared_gb: Optional[float] = None
    token: Optional[str] = None
    expected_revision: Optional[int] = None


class ComfySavedWorkflowRequest(BaseModel):
    name: str
    description: str = ""
    workflow: Any
    bindings: Dict[str, Any] = {}
    is_default: bool = False
    expected_revision: Optional[int] = None


class ComfyRunRequest(BaseModel):
    project_id: str
    shot_id: Optional[str] = None
    workflow: Any
    overrides: Dict[str, Dict[str, Any]] = {}
    filename_prefix: Optional[str] = None
    prompt: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    bindings: Dict[str, Any] = {}
    estimate_gb: Optional[float] = None
    estimate_seconds: Optional[float] = None
    timeout_sec: Optional[float] = None

class BoardNode(BaseModel):
    id: str
    x: float
    y: float
    width: float = 300
    height: float = 230
    pinned: bool = False


class BoardArea(BaseModel):
    id: str
    title: str
    x: float
    y: float
    width: float
    height: float
    color: Optional[str] = None
    collapsed: bool = False


class BoardLink(BaseModel):
    id: str
    source: str
    target: str
    kind: str = "thread"


class BoardLayoutPayload(BaseModel):
    nodes: List[BoardNode] = []
    areas: List[BoardArea] = []
    links: List[BoardLink] = []
    revision: Optional[int] = None


class BoardViewPayload(BaseModel):
    camera: Optional[dict] = None
    selection: List[str] = []
    surface: Optional[str] = None
    zoom: Optional[float] = None


class RenderSettings(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    fit: Literal["pad", "crop"] = "pad"
    crf: int = 18
    preset: str = "fast"
    audio_bitrate: str = "192k"
    audio_normalize: bool = False
    lut: Optional[str] = None


class RenderRequest(BaseModel):
    settings: RenderSettings = RenderSettings()
    track_states: Dict[str, dict] = {}
    source_path: Optional[str] = None  # explicit source for shorts/variants


class SelectTakeRequest(BaseModel):
    job_id: str


class FavoriteTakeRequest(BaseModel):
    job_id: Optional[str] = None


class ShareCommentRequest(BaseModel):
    shot_id: Optional[str] = None
    frame: int = 0
    body: str
    author: str = "reviewer"


class ShareApprovalRequest(BaseModel):
    shot_id: str
    status: Literal["approved", "changes_requested"]
    note: str = ""
    author: str = "reviewer"


class ReviewCommentRequest(BaseModel):
    shot_id: Optional[str] = None
    frame: int = 0
    body: str
    author: str = "local"


class ReviewCommentUpdate(BaseModel):
    body: Optional[str] = None
    status: Optional[Literal["open", "resolved"]] = None


class ReviewApprovalRequest(BaseModel):
    shot_id: str
    status: Literal["approved", "changes_requested"]
    note: str = ""
    author: str = "local"


class DepthRequest(BaseModel):
    image_path: str
    shot_id: Optional[str] = None


class PlanRequest(BaseModel):
    text: str
    mode: str = "direct"  # ask | direct | generate | find
    shot_id: Optional[str] = None


class PlanStep(BaseModel):
    kind: str
    label: str
    target: Optional[str] = None
    before: Optional[Any] = None
    after: Optional[Any] = None
    detail: Optional[str] = None


class PlanEstimate(BaseModel):
    model: Optional[str] = None
    gpu_seconds: Optional[float] = None
    vram_mb: Optional[int] = None
    note: str = "Rough local estimate; the queue reports actual time."


class PlanResponse(BaseModel):
    mode: str
    summary: str
    steps: List[PlanStep] = []
    estimate: Optional[PlanEstimate] = None
    warnings: List[str] = []
    shot_id: Optional[str] = None


class TrackingSaveRequest(BaseModel):
    session_id: str
    video_path: Optional[str] = None
    frames: List[dict]  # List of {frame_idx, num_objects, masks: {obj_id: b64}, scores: {obj_id: float}}
    objects: Optional[dict] = None  # Metadata: {obj_id: {label: str, color: str}}
