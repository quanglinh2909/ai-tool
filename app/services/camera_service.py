import json
import time

from piccolo.columns import UUID

from app.constants.type_ai_enum import TypeAIEnum
from app.dto.camera_dto import CameraDTO
from app.models.camera_model import Camera
from onvif import ONVIFCamera

from app.models.process_model import ProcessAI
from app.services.ai_plate_service import ai_plate_service
from app.ultils.ultils import get_rtsp_encode


class CameraService:
    async def create(self, req: CameraDTO):
        # Lưu camera
        camera = Camera(req.model_dump(exclude={"setting"}))
        camera = await camera.save()

        # Nếu bật nhận diện biển số, lưu ProcessAI
        setting = req.setting
        if setting and setting.is_detect_plate:
            id = camera[0].get("id")
            process_ai = ProcessAI(
                camera=camera[0].get("id"),
                direction=setting.direction_angle_plate,
                points=setting.points_plate,  # Make sure this is JSON serializable
                type_ai=TypeAIEnum.PLATE
            )
            await process_ai.save()
            rtsp =get_rtsp_encode(req.rtsp, req.username, req.password)
            ai_plate_service.add_camera(id,rtsp,req.setting.points_plate,int(setting.direction_angle_plate))

        return camera

    async def update(self,req: CameraDTO,camera_id: UUID):
        camera = await Camera.objects().get(Camera.id == camera_id).first()

        for key, value in req.model_dump(exclude_unset=True).items():
            setattr(camera, key, value)

        await camera.save()
        return camera



    def get_rtsp(self,req: CameraDTO):
            try:
                camera = ONVIFCamera(req.ip, port=req.htt_port, user=req.username, passwd=req.password)
                media_service = camera.create_media_service()
                profiles = media_service.GetProfiles()
                profile_token = profiles[0].token

                stream_setup = {
                    'Stream': 'RTP-Unicast',
                    'Transport': {'Protocol': 'RTSP'}
                }

                stream_uri = media_service.GetStreamUri({'StreamSetup': stream_setup, 'ProfileToken': profile_token})

                # bo &unicast=true&proto=Onvif
                uri = stream_uri.Uri.replace("&unicast=true&proto=Onvif", "")

                # print("RTSP URL:", stream_uri.Uri)
                return {"status_code": 200, "rtsp": uri}

            except Exception as e:
                print(f"Lỗi xảy ra: {e}")
                return {"status_code": 500, "message": str(e)}

    async def get_all_cameras(self):
        cameras = await Camera.select()
        return cameras




camera_service = CameraService()