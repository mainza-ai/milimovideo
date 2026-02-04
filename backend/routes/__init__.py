from fastapi import APIRouter
from .projects import router as projects_router
from .jobs import router as jobs_router
from .assets import router as assets_router
from .storyboard import router as storyboard_router
from .elements import router as elements_router

router = APIRouter()

router.include_router(projects_router)
router.include_router(jobs_router)
router.include_router(assets_router)
router.include_router(storyboard_router)
router.include_router(elements_router)

