from typing import Any

from fastapi import WebSocket
import cv2
import asyncio

from app.constants.platform_enum import PlatformEnum
from app.ultils.camera_ultil import get_rtsp_platform
from app.ultils.check_platform import get_os_name


class ConnectionManager:
    active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, rtsp):
        await websocket.accept()
        if rtsp in self.active_connections:
            self.active_connections[rtsp].append(websocket)
        else:
            self.active_connections[rtsp] = [websocket]
            asyncio.create_task(self.stream_camera(rtsp))




    async def disconnect(self, websocket: WebSocket):
        for key, value in list(self.active_connections.items()):
            if websocket in value:
                value.remove(websocket)
                if not value:
                    del self.active_connections[key]
                break

    async def send_video(self, rtsp: str, data):
        if rtsp in self.active_connections:
            for connection in self.active_connections[rtsp]:
                try:
                    await connection.send_bytes(data)
                except Exception as e:
                    print("send_company_message_json", e)
    async def send_json(self,rtsp, message: dict):
        if rtsp in self.active_connections:
            for connection in self.active_connections[rtsp]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print("send_company_message_json", e)

    async def stream_camera(self,rtsp):
        platform = get_os_name()
        tsp = get_rtsp_platform(rtsp, platform)
        cap = cv2.VideoCapture(tsp)
        while rtsp in self.active_connections:
            ret, frame = cap.read()
            if not ret:
                await self.send_json(rtsp, {"error": "Failed to read frame"})
                break
            frame = cv2.resize(frame, (640, 480))
            if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame)
            data = buffer.tobytes()
            await self.send_video(rtsp, data)
            await asyncio.sleep(0.01)  # Giảm tải CPU



        cap.release()
        print("Camera stream stopped for", rtsp)



connection_manager = ConnectionManager()
