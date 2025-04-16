import time

import cv2
import base64
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from multiprocessing import Process, Queue, set_start_method

app = FastAPI()
frame_queue = Queue(maxsize=1)


def capture_frames(queue):
    cap = cv2.VideoCapture("rtsp://admin:Oryza123@192.168.104.2:554/cam/realmonitor?channel=1&subtype=0")
    # cap = cv2.VideoCapture("output_vao.avi")
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return

    print("✅ Đã mở camera thành công, đang bắt đầu thu thập frame...")
    try:
        while True:
            ret, frame = cap.read()
            # time.sleep(1)
            if not ret:
                print("⚠️ Không đọc được frame từ camera")
                continue

            # Xóa frame cũ nếu queue đầy
            if queue.full():
                try:
                    queue.get_nowait()
                except:
                    pass

            # Thêm frame mới vào queue
            queue.put(frame)
    except Exception as e:
        print(f"❌ Lỗi khi đọc camera: {str(e)}")
    finally:
        cap.release()
        print("📷 Đã đóng camera")


@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client đã kết nối WebSocket")

    try:
        while True:
            frame = None
            # Lấy frame mới nhất từ queue
            while not frame_queue.empty():
                try:
                    frame = frame_queue.get_nowait()
                except:
                    continue

            if frame is not None:
                # Encode frame thành JPEG
                frame = cv2.resize(frame, (640, 480))  # Resize frame nếu cần
                ret, buffer = cv2.imencode('.jpg', frame)
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
    p = Process(target=capture_frames, args=(frame_queue,))
    p.start()
    print("✅ Đã khởi động process thu thập video")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8654)
    finally:
        print("🛑 Đang dừng server...")
        p.terminate()
        p.join()
        print("✅ Đã dừng process thu thập video")