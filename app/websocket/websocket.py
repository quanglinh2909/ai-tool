import asyncio
import json
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
    count_client_incremented = False
    try:
        camera_id = UUID(camera_id)
        if camera_id not in ai_plate_service.shared_memories:
            await websocket.close(code=1008, reason="Camera ID không hợp lệ hoặc không tồn tại.")
            return
        data = ai_plate_service.shared_memories[camera_id]
        print("🔌 Client đã kết nối WebSocket", data)
        shm = data.get("shm")
        shape = data.get("shape")
        dtype = data.get("dtype")
        ready_event = data.get("ready_event")
        count_client = data.get("count_client")
        frame_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        # set count_client
        count_client.value += 1
        count_client_incremented = True
        target_fps = 15
        last_frame_time = 0
        frame_delay = 1.0 / target_fps

        while True:
            # Phiên bản không chặn của việc đợi event
            # Thêm timeout để tránh treo vô hạn
            wait_start_time = time.time()
            max_wait_time = 5.0  # 5 giây timeout

            while not ready_event.is_set():
                # Kiểm tra nếu đã đợi quá lâu
                if time.time() - wait_start_time > max_wait_time:
                    print("⚠️ Timeout đợi frame mới từ camera")
                    # Gửi thông báo lỗi hoặc frame trống cho client
                    await websocket.send_text(json.dumps({"error": "Camera timeout"}))
                    await asyncio.sleep(1)  # Đợi 1 giây trước khi thử lại
                    break
                await asyncio.sleep(0.01)  # Sleep ngắn để không chặn event loop

            # Nếu chờ quá lâu thì tiếp tục vòng lặp
            if time.time() - wait_start_time > max_wait_time:
                continue

            frame_copy = frame_np.copy()
            ready_event.clear()  # Reset cờ

            current_time = time.time()
            time_elapsed = current_time - last_frame_time

            if time_elapsed < frame_delay:
                await asyncio.sleep(frame_delay - time_elapsed)
                continue

            last_frame_time = time.time()

            if frame_copy is not None:
                ret, buffer = cv2.imencode('.jpg', frame_copy)
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("⚠️ WebSocket bị ngắt kết nối")
    except Exception as e:
        print(f"❌ Lỗi WebSocket: {str(e)}")
    finally:
        # Đảm bảo giảm counter cho dù có lỗi xảy ra
        if count_client_incremented and 'count_client' in locals():
            count_client.value -= 1