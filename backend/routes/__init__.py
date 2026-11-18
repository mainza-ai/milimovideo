from fastapi import APIRouter
from .projects import router as projects_router
from .jobs import router as jobs_router
from .assets import router as assets_router
from .storyboard import router as storyboard_router
from .elements import router as elements_router
from .shots import router as shots_router
from .models import router as models_router
from .planner import router as planner_router
from .review import router as review_router
from .share import router as share_router
from .depth import router as depth_router
from .settings import router as settings_router
from .memory import router as memory_router
from .comfy import router as comfy_router

router = APIRouter()

router.include_router(projects_router)
router.include_router(jobs_router)
router.include_router(assets_router)
router.include_router(storyboard_router)
router.include_router(elements_router)
router.include_router(shots_router)
router.include_router(models_router)
router.include_router(planner_router)
router.include_router(review_router)
router.include_router(share_router)
router.include_router(depth_router)
router.include_router(settings_router)
router.include_router(memory_router)
router.include_router(comfy_router)

