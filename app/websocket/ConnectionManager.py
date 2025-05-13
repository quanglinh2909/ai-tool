import queue
import threading
import time
from typing import Any

import av
from fastapi import WebSocket
import cv2
import asyncio

from app.constants.platform_enum import PlatformEnum
from app.ultils.camera_ultil import get_rtsp_platform, decode_frames
from app.ultils.check_platform import get_os_name


class ConnectionManager:
    active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, rtsp,w,h):
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
    def call_back_error(self,rtsp,status_code, mess):
        asyncio.run(self.send_json(rtsp, {"error": "Failed to read frame"}))

    async def stream_camera(self,rtsp):
        max_size_queue = 1
        frame_queue = queue.Queue(maxsize=max_size_queue)
        stop_event = threading.Event()
        decode_thread = threading.Thread(
            target=decode_frames,
            args=(rtsp, frame_queue, stop_event, max_size_queue,self.call_back_error),
        )
        decode_thread.start()

        target_fps = 15

        # Biến theo dõi thời gian cho việc giới hạn FPS
        last_frame_time = 0
        frame_delay = 1.0 / target_fps  # Tính toán thời gian trễ giữa các frame
        print("Starting camera stream for", rtsp)

        while not stop_event.is_set() and rtsp in self.active_connections:
            try:
                # print("Đang lấy frame từ camera")
                # Tính thời gian cần thiết để đạt được FPS mục tiêu
                current_time = time.time()
                time_elapsed = current_time - last_frame_time

                # Nếu chưa đến thời gian cần lấy frame tiếp theo, sleep đi một chút
                if time_elapsed < frame_delay:
                    await asyncio.sleep(frame_delay - time_elapsed)
                    continue

                # Ghi lại thời điểm lấy frame
                last_frame_time = time.time()

                # Lấy frame từ queue với timeout
                frame = frame_queue.get(timeout=1.0)
                # resized_frame = frame.reformat( format="bgr24")
                resized_frame = frame.reformat(width=640, height=480, format="bgr24")
                frame = resized_frame.to_ndarray()
                _, buffer = cv2.imencode('.jpg', frame)
                data = buffer.tobytes()
                await self.send_video(rtsp, data)
                await asyncio.sleep(0.001)
            except queue.Empty:
                continue

        stop_event.set()
        print("Camera stream stopped for", rtsp)

        # while rtsp in self.active_connections:
        #     ret, frame = cap.read()
        #     if not ret:
        #         await self.send_json(rtsp, {"error": "Failed to read frame"})
        #         break
        #     frame = cv2.resize(frame, (640, 480))
        #     if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     if w is None or h is None:
        #         w = frame.shape[1]
        #         h = frame.shape[0]
        #         await self.send_json(rtsp, {
        #             "type": "info",
        #             "width": w,
        #             "height": h
        #         })
        #
        #     _, buffer = cv2.imencode('.jpg', frame)
        #     data = buffer.tobytes()
        #     await self.send_video(rtsp, data)
        #     await asyncio.sleep(0.01)  # Giảm tải CPU
        #
        #
        # cap.release()
        # print("Camera stream stopped for", rtsp)



connection_manager = ConnectionManager()
