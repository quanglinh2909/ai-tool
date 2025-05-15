from fastapi import APIRouter

from app.routers.camera_router import router as camera_router
from app.routers.process_router import router as process_router

api_router = APIRouter()

api_router.include_router(camera_router, prefix="/camera", tags=["Camera"])
api_router.include_router(process_router, prefix="/process", tags=["Process"])

