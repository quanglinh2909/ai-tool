import atexit
import multiprocessing
import queue
import signal
import os
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from app.constants.platform_enum import PlatformEnum
from app.ultils.camera_ultil import get_model_plate_platform, decode_frames
from app.ultils.centroid_tracker import CentroidTracker, draw_tracks
from app.ultils.check_platform import get_os_name
from app.ultils.coco_utils import COCO_test_helper
from app.ultils.drraw_image import draw_identification_area, draw_direction_vector, draw_box
from app.ultils.post_process import setup_model, post_process

import os

from app.ultils.ultils import point_in_polygon

os.environ['RKNN_LOG_LEVEL'] = '0'
class AIPlateService:
    def __init__(self):
        self.processes = {}
        self.rtsps = {}
        self.shared_boxes = {}
        self.shared_memories = {}
        self.shared_angles = {}
        self.shared_send_frame = {}
        self.stt = 0
        self.stop_event = multiprocessing.Event()  # Sự kiện dừng toàn cục

        # Đăng ký hàm cleanup khi thoát
        atexit.register(self.cleanup_all)

        # Đăng ký xử lý tín hiệu
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Khởi động thread giám sát quy trình
        self.monitor_thread = threading.Thread(target=self.check_processes, daemon=True)
        self.monitor_thread.start()

    def signal_handler(self, sig, frame):
        """Xử lý tín hiệu tắt từ hệ thống"""
        print(f"Nhận tín hiệu {sig}, đang dọn dẹp...")
        self.cleanup_all()
        os._exit(0)  # Buộc thoát

    def cleanup_all(self):
        """Dọn dẹp tất cả tài nguyên trước khi thoát"""
        print("Đang dọn dẹp tất cả tài nguyên...")
        self.stop_event.set()  # Báo hiệu cho tất cả các worker dừng lại

        # Dừng tất cả các process
        for camera_id in list(self.processes.keys()):
            self.remove_camera(camera_id)

        print("Đã dọn dẹp xong tất cả tài nguyên")

    def worker(self, stt, camera_id, rtsp, shared_array, angle_shared, count_client, shm_name, shape, dtype,
               ready_event,
               is_show=True):
        # Thiết lập xử lý tín hiệu cho worker process
        signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))
        signal.signal(signal.SIGINT, lambda sig, frame: exit(0))

        # Còn lại giữ nguyên...
        CLASSES = ("plate")
        IMG_SIZE = (640, 640)
        tracker = CentroidTracker(max_disappeared=20, max_distance=50)
        co_helper = COCO_test_helper(enable_letter_box=True)
        platform = get_os_name()
        args = get_model_plate_platform(platform)
        args["stt"] = stt
        print("args", args)
        model, _platform = setup_model(args)

        max_size_queue = 1
        # Thiết lập threading
        frame_queue = queue.Queue(maxsize=max_size_queue)
        stop_event = threading.Event()

        existing_shm = shared_memory.SharedMemory(name=shm_name)
        frame_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        # Thread giải mã frame
        decode_thread = threading.Thread(
            target=decode_frames,
            args=(rtsp, frame_queue, stop_event, max_size_queue),
        )
        decode_thread.daemon = True
        decode_thread.start()

        # Kiểm tra sự kiện dừng toàn cục
        frame_count = 0
        start_time = time.time()
        fps = 0
        target_fps = 15

        # Biến theo dõi thời gian cho việc giới hạn FPS
        last_frame_time = 0
        frame_delay = 1.0 / target_fps  # Tính toán thời gian trễ giữa các frame

        while not stop_event.is_set() and not self.stop_event.is_set():
            try:
                client_count = count_client.value
                data = np.array(shared_array).reshape(-1, 2)
                angle = angle_shared.value

                # Tính thời gian cần thiết để đạt được FPS mục tiêu
                current_time = time.time()
                time_elapsed = current_time - last_frame_time

                # Nếu chưa đến thời gian cần lấy frame tiếp theo, sleep đi một chút
                if time_elapsed < frame_delay:
                    time.sleep(frame_delay - time_elapsed)
                    continue

                # Ghi lại thời điểm lấy frame
                last_frame_time = time.time()

                # Lấy frame từ queue với timeout
                frame = frame_queue.get(timeout=1.0)
                frame = frame.reformat(format="bgr24")
                frame = frame.to_ndarray()

                img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                                           pad_color=(0, 0, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if platform != PlatformEnum.ORANGE_PI_MAX and platform != PlatformEnum.ORANGE_PI:
                    img = np.transpose(img, (2, 0, 1))  # (3, 640, 640)
                    img = img.astype(np.float32)
                img = np.expand_dims(img, axis=0)

                outputs = model.run([img])
                boxes, classes, scores = post_process(outputs)

                roi_points_drawn, roi_points = draw_identification_area(data, frame, is_draw=True)
                arrow_vector = draw_direction_vector(roi_points_drawn, frame, angle, is_draw=True)

                if boxes is not None and len(boxes) > 0:
                    # Chuyển đổi boxes về tọa độ thực
                    real_boxes = co_helper.get_real_box(boxes)

                    for box, score, cl in zip(real_boxes, scores, classes):
                        x, y, w, h = [int(_b) for _b in box]
                        draw_box(frame, x, y, w, h, is_draw=True)

                    # # Cập nhật tracker
                    # tracks = tracker.update(real_boxes, scores, classes)
                    # # Vẽ các track lên frame
                    # frame = draw_tracks(frame, tracks, CLASSES)

                # Tính FPS thực tế để hiển thị
                frame_count += 1
                elapsed_time = time.time() - start_time

                # Cập nhật FPS sau mỗi giây
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    print('{} - FPS: {:.2f} (target: {})'.format(camera_id, fps, target_fps))
                    frame_count = 0
                    start_time = time.time()

                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)





                if client_count > 0:
                    frame = cv2.resize(frame, (640, 480))
                    np.copyto(frame_np, frame)
                    # Đánh dấu là đã sẵn sàng
                    ready_event.set()

                time.sleep(0.001)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                break

        # Dọn dẹp khi worker kết thúc
        stop_event.set()
        if decode_thread.is_alive():
            decode_thread.join(timeout=1.0)
        try:
            existing_shm.close()
        except Exception as e:
            print(f"Error closing shared memory in worker: {e}")
        print(f"Worker process for camera {camera_id} exiting")

    def add_camera(self, camera_id, rtsp, points, angle):
        """
        Thêm camera mới để theo dõi

        Args:
            camera_id: ID của camera
            rtsp: URL RTSP của camera
            points: Danh sách các điểm định nghĩa ROI
            angle: Góc của mũi tên chỉ hướng
        """
        if camera_id in self.processes:
            print(f"Camera {camera_id} đã đang chạy.")
            return False

        try:

            # Chuyển đổi dữ liệu thành mảng numpy
            shared_data = np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)
            shared_array = multiprocessing.Array('d', shared_data.flatten())  # 'd' là kiểu float64

            # cap = cv2.VideoCapture(rtsp)
            # if not cap.isOpened():
            #     print(f"[ERROR] Không thể mở camera {camera_id} với URL {rtsp}")
            #     return False
            # # Lấy kích thước của frame
            # ret, frame = cap.read()
            # if not ret:
            #     print(f"[ERROR] Không thể đọc frame từ camera {camera_id}")
            #     return False
            # height, width, _ = frame.shape
            # cap.release()


            # Cấu hình shared memory cho frame
            # shape = (height, width, 3)  # height, width, channels
            shape = (480  , 640, 3)  # height, width, channels
            dtype = np.uint8
            size = int(np.prod(shape))  # 640 * 480 * 3 = 921600

            try:
                shm_global = shared_memory.SharedMemory(create=True, size=size)
                ready_event = multiprocessing.Event()

                # Tạo biến chia sẻ cho góc và số client
                angle_shared = multiprocessing.Value('i', angle)
                count_client = multiprocessing.Value('i', 0)

                # Khởi động worker process
                process = multiprocessing.Process(
                    target=self.worker,
                    args=(self.stt,camera_id, rtsp, shared_array, angle_shared, count_client,
                          shm_global.name, shape, dtype, ready_event),
                    daemon=True
                )
                process.start()

                # Lưu thông tin vào các dictionary
                self.processes[camera_id] = process
                self.rtsps[camera_id] = rtsp
                self.shared_boxes[camera_id] = shared_array
                self.shared_memories[camera_id] = {
                    "shm": shm_global,
                    "shape": shape,
                    "dtype": dtype,
                    "ready_event": ready_event,
                    "count_client": count_client,
                }
                self.shared_angles[camera_id] = angle_shared
                self.shared_send_frame[camera_id] = count_client
                self.stt += 1

                print(f"Đã thêm camera {camera_id} thành công")
                return True

            except Exception as e:
                print(f"[ERROR] Không thể tạo shared memory: {e}")
                return False

        except Exception as e:
            print(f"[ERROR] Không thể thêm camera {camera_id}: {e}")
            return False

    def remove_camera(self, id_camera):
        if id_camera in self.processes:
            # Dừng và dọn dẹp process
            try:
                self.processes[id_camera].terminate()
                self.processes[id_camera].join(timeout=3)
                if self.processes[id_camera].is_alive():
                    print(f"Process for camera {id_camera} still alive after terminate, killing...")
                    self.processes[id_camera].kill()  # Buộc dừng nếu không thể terminate
                    # Đảm bảo process đã dừng hoàn toàn
                    self.processes[id_camera].join(timeout=1)
                del self.processes[id_camera]
                print(f"Đã dừng camera {id_camera}.")
            except Exception as e:
                print(f"[ERROR] Lỗi khi dừng process cho camera {id_camera}: {e}")

        # Dọn dẹp shared memory
        if id_camera in self.shared_memories:
            try:
                memory_info = self.shared_memories[id_camera]
                memory_info["shm"].close()
                memory_info["shm"].unlink()  # Quan trọng: giải phóng shared memory
                del self.shared_memories[id_camera]
            except Exception as e:
                print(f"[ERROR] Lỗi khi giải phóng shared memory cho camera {id_camera}: {e}")

        # Dọn dẹp các dữ liệu khác
        # ...

        return True

    def check_processes(self):
        """Thread kiểm tra và khởi động lại các process đã bị dừng"""
        while not self.stop_event.is_set():
            try:
                for cam_id, p in list(self.processes.items()):
                    if not p.is_alive():
                        print(f"[WARN] Process cho camera {cam_id} đã dừng. Đang khởi động lại...")
                        # Mã khởi động lại camera...
            except Exception as e:
                print(f"[ERROR] Lỗi trong thread kiểm tra: {e}")

            time.sleep(5)  # Kiểm tra mỗi 5 giây
        print("Monitor thread exiting")


# Tạo instance mặc định với cách xử lý thoát an toàn
ai_plate_service = AIPlateService()


