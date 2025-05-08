import json

from app.constants.type_ai_enum import TypeAIEnum
from app.models.process_model import ProcessAI
from app.services.ai_plate_service import ai_plate_service
from app.ultils.ultils import get_rtsp_encode


class ProcessAiService:
    async def init_porcess_ai(self):
        process_aies = await ProcessAI.select(ProcessAI.camera.rtsp,
                                              ProcessAI.camera.id,
                                              ProcessAI.camera.username,
                                              ProcessAI.camera.password,
                                              ProcessAI.direction,
                                              ProcessAI.points,
                                              ProcessAI.type_ai)
        for process_ai in process_aies:
           type_ai = process_ai.get('type_ai')
           points = process_ai.get('points')
           if type_ai == TypeAIEnum.PLATE.value:
                rtsp = process_ai.get('camera_id.rtsp')
                id = process_ai.get('camera_id.id')
                username = process_ai.get('camera_id.username')
                password = process_ai.get('camera_id.password')
                rtsp = get_rtsp_encode(rtsp, username, password)
                print("points",type(points))
                points = json.loads(points)
                ai_plate_service.add_camera(id, rtsp,points)


process_ai_service = ProcessAiService()
