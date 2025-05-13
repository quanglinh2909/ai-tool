import asyncio
import time
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Query
from starlette.websockets import WebSocket, WebSocketDisconnect
from typing import List

from app.services.ai_plate_service import ai_plate_service
from app.websocket.ConnectionManager import connection_manager

router = APIRouter()


@router.websocket("/stream")
async def websocket_super_admin(websocket: WebSocket,rtsp,w,h):

    await connection_manager.connect(websocket, rtsp,w,h)
    try:
        while True:
            data = await websocket.receive_json()
            # Xử lý các action khác nếu cần
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)

@router.websocket("/camera")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    print("🔌 Client đã kết nối WebSocket")

    try:
        camera_id = UUID(camera_id)
        if camera_id not in ai_plate_service.shared_memories:
            # await websocket.close(code=1008, reason="Camera ID không hợp lệ hoặc không tồn tại.")
            return
        data = ai_plate_service.shared_memories[camera_id]
        print("🔌 Client đã kết nối WebSocket",data)
        shm = data.get("shm")
        shape = data.get("shape")
        dtype = data.get("dtype")
        ready_event = data.get("ready_event")
        count_client = data.get("count_client")
        frame_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        # set count_client
        count_client.value += 1

        target_fps = 15

        # Biến theo dõi thời gian cho việc giới hạn FPS
        last_frame_time = 0
        frame_delay = 1.0 / target_fps  # Tính toán thời gian trễ giữa các frame
        while True:
            # print("🔌 Đang chờ frame mới từ camera...")
            ready_event.wait()  # Đợi cho đến khi có frame mới
            # print("🔌 Đã nhận frame mới từ camera")

            frame_copy = frame_np.copy()  # Copy ra riêng để tránh xung đột
            ready_event.clear()  # Reset cờ
            current_time = time.time()
            time_elapsed = current_time - last_frame_time

            # Nếu chưa đến thời gian cần lấy frame tiếp theo, sleep đi một chút
            if time_elapsed < frame_delay:
                await asyncio.sleep(frame_delay - time_elapsed)
                continue

            if frame_copy is not None:
                # Encode frame thành JPEG
                ret, buffer = cv2.imencode('.jpg', frame_copy)
                # Gửi dữ liệu binary
                await websocket.send_bytes(buffer.tobytes())

            # Tạm dừng để không làm quá tải CPU
            await asyncio.sleep(0.01)  # 100 FPS cap (thực tế sẽ thấp hơn do thời gian encode)

    except WebSocketDisconnect:
        print("⚠️ WebSocket bị ngắt kết nối")
        # Giảm số lượng client đang kết nối
        count_client.value -= 1
    except Exception as e:
        print(f"❌ Lỗi WebSocket: {str(e)}")
