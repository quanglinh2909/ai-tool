from piccolo.columns import UUID

from app.dto.camera_dto import CameraDTO
from app.models.camera_model import Camera


class CameraService:
    async def create(self,req: CameraDTO):
        camera = Camera(req.model_dump())
        return await camera.save()

    async def update(self,req: CameraDTO,camera_id: UUID):
        camera = await Camera.objects().get(Camera.id == camera_id).first()

        for key, value in req.model_dump(exclude_unset=True).items():
            setattr(camera, key, value)

        await camera.save()
        return camera


camera_service = CameraService()