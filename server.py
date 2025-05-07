import asyncio
import time
from multiprocessing import Process, shared_memory, Event

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

shape = (480, 640, 3)  # height, width, channels
dtype = np.uint8
size = np.prod(shape)  # 640 * 480 * 3 = 921600

# Tạo vùng shared memory
shm_global = shared_memory.SharedMemory(create=True, size=size)
ready_event = Event()

def capture_frames(shm_name, shape, dtype, ready_event):
    cap = cv2.VideoCapture("rtsp://admin:Oryza123@192.168.104.2:554/cam/realmonitor?channel=1&subtype=0")
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize hoặc crop nếu frame không đúng kích thước
        frame = cv2.resize(frame, (shape[1], shape[0]))

        # Ghi frame vào shared memory
        np.copyto(frame_np, frame)

        # Đánh dấu là đã sẵn sàng
        ready_event.set()
        time.sleep(0.01)

    cap.release()
    existing_shm.close()


@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client đã kết nối WebSocket")

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


if __name__ == "__main__":
    # set_start_method("spawn", force=True)
    print("🚀 Khởi động server video stream...")

    # Bắt đầu process để capture video
    reader = Process(target=capture_frames, args=(shm_global.shm_global, shape, dtype, ready_event))
    reader.start()

    print("✅ Đã khởi động process thu thập video")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8654)
    finally:
        print("🛑 Đang dừng server...")
        reader.terminate()
        reader.join()
        print("✅ Đã dừng process thu thập video")
