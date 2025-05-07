import asyncio
from contextlib import asynccontextmanager
from multiprocessing import shared_memory

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

import uvicorn
from fastapi import FastAPI

from app.app import api_router
from app.services.ai_plate_service import AIPlateService, ai_plate_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    ai_plate_service.add_camera(1, "rtsp://admin:Oryza123@192.168.104.2:554/cam/realmonitor?channel=1&subtype=0")
    ai_plate_service.add_camera(2, "rtsp://admin:Oryza123@192.168.104.108:554/cam/realmonitor?channel=1&subtype=0")

    print("Starting the server")
    yield
    print("Shutting down the server")


app = FastAPI(
    docs_url="/",
    root_path="/ai",
    openapi_url="/openapi.json",
    title="ai Service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc list origin cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/video/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    print("🔌 Client đã kết nối WebSocket")
    if camera_id not in ai_plate_service.shared_memories:
        print(f"❌ Không tìm thấy camera với ID {camera_id}")
        await websocket.close()
        return

    data  = ai_plate_service.shared_memories[camera_id]
    shape = data["shape"]
    dtype = data["dtype"]
    shm_global = data["shm"]
    ready_event = data["ready_event"]


    try:
        shm = shared_memory.SharedMemory(name=shm_global.name)
        frame_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        while True:
            ready_event.wait()  # Đợi cho đến khi có frame mới

            frame_copy = frame_np.copy()  # Copy ra riêng để tránh xung đột
            ready_event.clear()  # Reset cờ

            if frame_copy is not None:
                # Encode frame thành JPEG
                ret, buffer = cv2.imencode('.jpg', frame_copy)
                # Gửi dữ liệu binary
                await websocket.send_bytes(buffer.tobytes())

            # Tạm dừng để không làm quá tải CPU
            await asyncio.sleep(0.01)  # 100 FPS cap (thực tế sẽ thấp hơn do thời gian encode)

    except WebSocketDisconnect:
        print("⚠️ WebSocket bị ngắt kết nối")
    except Exception as e:
        print(f"❌ Lỗi WebSocket: {str(e)}")


app.include_router(api_router)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8007)
