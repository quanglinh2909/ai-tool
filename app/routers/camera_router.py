from fastapi import APIRouter, Depends

from app.dto.camera_dto import DrawBoundingBoxDTO
from app.services.ai_plate_service import ai_plate_service

router = APIRouter()

@router.post("/add-camera")
async def add_camera():
    return "Camera added successfully"

@router.delete("/remove-camera")
async def remove_camera():
    return "Camera removed successfully"

# ve vung nhan dien
@router.post("/draw-bounding-box")
def draw_bounding_box(new_data:DrawBoundingBoxDTO):
    print(new_data)

    return ai_plate_service.update_shared_array(new_data.camera_id,new_data.data,new_data.angle)
