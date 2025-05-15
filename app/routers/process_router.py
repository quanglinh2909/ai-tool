from uuid import UUID

from fastapi import APIRouter, Depends

from app.services.process_ai_service import process_ai_service

router = APIRouter()

@router.get("/get-by-id-camera/{camera_id}")
async def get_camera_by_id(camera_id: UUID):
    """
    Get camera by id
    """
    return await process_ai_service.get_process_ai_by_camera_id(camera_id)



