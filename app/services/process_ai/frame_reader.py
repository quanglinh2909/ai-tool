import queue
import time
import cv2

from app.constants.platform_enum import PlatformEnum


def frame_reader( rtsp, frame_queue, stop_event, platform):
    """
    Luồng chuyên đọc khung hình từ camera và cập nhật vào queue

    Args:
        rtsp: URL RTSP của camera
        frame_queue: Queue chứa khung hình mới nhất
        stop_event: Event để dừng luồng
        platform: Nền tảng phần cứng (để xử lý định dạng màu)
    """
    reconnect_delay = 5  # Thời gian chờ kết nối lại

    while not stop_event.is_set():
        try:
            print(f"[INFO] Đang kết nối tới: {rtsp}")
            cap = cv2.VideoCapture(rtsp)
            if not cap.isOpened():
                print(f"[WARN] Không thể kết nối. Thử lại sau {reconnect_delay} giây...")
                time.sleep(reconnect_delay)
                continue

            print("[INFO] Đã kết nối với camera.")
            fps_limit = 30  # Giới hạn FPS để tránh quá tải
            frame_time = 1.0 / fps_limit
            last_frame_time = 0

            while not stop_event.is_set() and cap.isOpened():
                try:
                    current_time = time.time()
                    # Giới hạn tốc độ đọc khung hình
                    if current_time - last_frame_time < frame_time:
                        time.sleep(0.001)  # Ngủ ngắn để giảm tải CPU
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        print("[WARN] Mất frame. Đang kết nối lại...")
                        break

                    last_frame_time = current_time
                    frame = cv2.resize(frame, (640, 480))

                    # Chuyển đổi màu sắc từ BGR sang RGB nếu cần
                    if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

                    # Cập nhật khung hình mới nhất vào queue
                    try:
                        frame_queue.put(frame.copy(), block=False)
                    except queue.Full:
                        try:
                            # Lấy một frame ra để có chỗ cho frame mới
                            frame_queue.get_nowait()
                            # Thêm khung hình mới
                            frame_queue.put(frame.copy(), block=False)
                        except Exception as e:
                            print(f"[ERROR] Lỗi khi cập nhật queue: {e}")

                except Exception as e:
                    print(f"[ERROR] Lỗi trong vòng lặp đọc khung hình: {e}")
                    time.sleep(0.1)

            cap.release()
            print(f"[INFO] Kết nối lại sau {reconnect_delay} giây...")
            time.sleep(reconnect_delay)

        except Exception as e:
            print(f"[ERROR] Lỗi trong frame_reader: {e}")
            time.sleep(1)
