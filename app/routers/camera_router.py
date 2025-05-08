from uuid import UUID

from fastapi import APIRouter, Depends

from app.dto.camera_dto import DrawBoundingBoxDTO, CameraDTO, CameraUpdateDTO, GetRtspDTO
from app.services.ai_plate_service import ai_plate_service
from app.services.camera_service import camera_service

router = APIRouter()

@router.post("/create")
async def create(req: CameraDTO):
    return await camera_service.create(req)

@router.put("/update/{camera_id}")
async def update(req: CameraUpdateDTO, camera_id: UUID):
    return await camera_service.update(req, camera_id)

@router.delete("/remove-camera")
async def remove_camera():
    return "Camera removed successfully"

@router.post("/get-rtsp")
def get_rtsp(req: GetRtspDTO):
    return  camera_service.get_rtsp(req)

@router.post("/draw-bounding-box")
def draw_bounding_box(new_data:DrawBoundingBoxDTO):
    print(new_data)

    return ai_plate_service.update_shared_array(new_data.camera_id,new_data.data,new_data.angle)
