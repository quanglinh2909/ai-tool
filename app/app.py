from fastapi import APIRouter

from app.routers.camera_router import router as camera_router

api_router = APIRouter()

api_router.include_router(camera_router, prefix="/camera", tags=["Camera"])

