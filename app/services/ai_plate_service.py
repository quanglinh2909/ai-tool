import multiprocessing
import queue
import threading
import time
from multiprocessing import shared_memory

import numpy as np

from app.services.process_ai.ai_processor import ai_processor
from app.services.process_ai.frame_reader import frame_reader
from app.ultils.camera_ultil import get_rtsp_platform
from app.ultils.check_platform import get_os_name


class AIPlateService:
    def __init__(self):
        self.processes = {}
        self.rtsps = {}
        self.shared_boxes = {}
        self.shared_memories = {}
        self.shared_angles = {}
        self.shared_send_frame = {}
        self.frame_process_modes = {}  # Lưu chế độ xử lý frame cho mỗi camera
        # Khởi động thread giám sát quy trình
        threading.Thread(target=self.check_processes, daemon=True).start()



    def worker(self, camera_id, rtsp, shared_array, angle_shared, count_client, shm_name, shape, dtype, ready_event,
               frame_process_mode=1, is_show=True):
        """
        Khởi động hai luồng riêng biệt cho việc đọc khung hình và xử lý AI

        Args:
            camera_id: ID của camera
            rtsp: URL RTSP của camera
            shared_array: Array chia sẻ cho các điểm ROI
            angle_shared: Giá trị chia sẻ cho góc mũi tên
            count_client: Biến đếm số client đang xem
            shm_name: Tên shared memory
            shape: Kích thước của frame
            dtype: Kiểu dữ liệu của frame
            ready_event: Event báo hiệu frame đã sẵn sàng
            frame_process_mode: Chế độ xử lý frame (1 = chỉ frame mới nhất, >1 = số lượng frame, 0 = tất cả)
            is_show: Hiển thị khung hình hay không
        """
        try:
            # Lấy thông tin nền tảng
            platform = get_os_name()

            # Chuẩn bị URL RTSP
            rtsp = get_rtsp_platform(rtsp, platform)

            # Tạo queue để chia sẻ khung hình giữa hai luồng
            # Điều chỉnh kích thước queue dựa trên chế độ xử lý
            if frame_process_mode <= 0:  # Chế độ xử lý tất cả
                queue_size = 100  # Kích thước lớn để lưu nhiều frame
            else:
                queue_size = max(5, frame_process_mode * 2)  # Đủ lớn cho số lượng frame cần xử lý

            frame_queue = queue.Queue(maxsize=queue_size)

            # Event để báo hiệu dừng các luồng
            stop_event = threading.Event()

            # Khởi động luồng đọc khung hình
            reader_thread = threading.Thread(
                target=frame_reader,
                args=(rtsp, frame_queue, stop_event, platform),
                daemon=True
            )
            reader_thread.start()

            # Khởi động luồng xử lý AI
            processor_thread = threading.Thread(
                target=ai_processor,
                args=(frame_queue, stop_event, camera_id, shared_array, angle_shared,
                      count_client, shm_name, shape, dtype, ready_event, frame_process_mode, is_show),
                daemon=True
            )
            processor_thread.start()

            # Đợi các luồng kết thúc
            while not stop_event.is_set():
                time.sleep(0.1)

                # Kiểm tra xem các luồng còn hoạt động không
                if not reader_thread.is_alive() or not processor_thread.is_alive():
                    print(f"[WARN] Một trong các luồng đã dừng. Thoát worker.")
                    stop_event.set()

            # Dọn dẹp
            if reader_thread.is_alive():
                reader_thread.join(timeout=2)
            if processor_thread.is_alive():
                processor_thread.join(timeout=2)

        except Exception as e:
            print(f"[ERROR] Lỗi trong worker: {e}")

    def add_camera(self, camera_id, rtsp, points, angle, frame_process_mode=50):
        """
        Thêm camera mới để theo dõi

        Args:
            camera_id: ID của camera
            rtsp: URL RTSP của camera
            points: Danh sách các điểm định nghĩa ROI
            angle: Góc của mũi tên chỉ hướng
            frame_process_mode: Chế độ xử lý frame (1 = chỉ frame mới nhất, >1 = số lượng frame, 0 = tất cả)
        """
        if camera_id in self.processes:
            print(f"Camera {camera_id} đã đang chạy.")
            return False

        try:
            # Chuyển đổi dữ liệu thành mảng numpy
            shared_data = np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)
            shared_array = multiprocessing.Array('d', shared_data.flatten())  # 'd' là kiểu float64

            # Cấu hình shared memory cho frame
            shape = (480, 640, 3)  # height, width, channels
            dtype = np.uint8
            size = int(np.prod(shape))  # 640 * 480 * 3 = 921600

            try:
                shm_global = shared_memory.SharedMemory(create=True, size=size)
                ready_event = multiprocessing.Event()

                # Tạo biến chia sẻ cho góc và số client
                angle_shared = multiprocessing.Value('i', angle)
                count_client = multiprocessing.Value('i', 0)

                # Lưu chế độ xử lý frame
                self.frame_process_modes[camera_id] = frame_process_mode

                # Khởi động worker process
                process = multiprocessing.Process(
                    target=self.worker,
                    args=(camera_id, rtsp, shared_array, angle_shared, count_client,
                          shm_global.name, shape, dtype, ready_event, frame_process_mode),
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
                    "ready_event": ready_event
                }
                self.shared_angles[camera_id] = angle_shared
                self.shared_send_frame[camera_id] = count_client

                print(f"Đã thêm camera {camera_id} thành công với chế độ xử lý frame: {frame_process_mode}")
                return True

            except Exception as e:
                print(f"[ERROR] Không thể tạo shared memory: {e}")
                return False

        except Exception as e:
            print(f"[ERROR] Không thể thêm camera {camera_id}: {e}")
            return False

    def remove_camera(self, id_camera):
        """
        Gỡ bỏ và dừng xử lý camera

        Args:
            id_camera: ID của camera cần gỡ bỏ
        """
        if id_camera in self.processes:
            # Dừng và dọn dẹp process
            try:
                self.processes[id_camera].terminate()
                self.processes[id_camera].join(timeout=3)
                if self.processes[id_camera].is_alive():
                    self.processes[id_camera].kill()  # Buộc dừng nếu không thể terminate
                del self.processes[id_camera]
                print(f"Đã dừng camera {id_camera}.")
            except Exception as e:
                print(f"[ERROR] Lỗi khi dừng process cho camera {id_camera}: {e}")

        # Dọn dẹp shared memory
        if id_camera in self.shared_memories:
            try:
                memory_info = self.shared_memories[id_camera]
                memory_info["shm"].close()
                memory_info["shm"].unlink()
                del self.shared_memories[id_camera]
            except Exception as e:
                print(f"[ERROR] Lỗi khi giải phóng shared memory cho camera {id_camera}: {e}")

        # Dọn dẹp các dữ liệu khác
        if id_camera in self.rtsps:
            del self.rtsps[id_camera]

        if id_camera in self.shared_boxes:
            del self.shared_boxes[id_camera]

        if id_camera in self.shared_angles:
            del self.shared_angles[id_camera]

        if id_camera in self.shared_send_frame:
            del self.shared_send_frame[id_camera]

        if id_camera in self.frame_process_modes:
            del self.frame_process_modes[id_camera]

        print(f"Đã xóa camera {id_camera} khỏi danh sách.")
        return True

    def update_shared_array(self, camera_id, new_data, angle):
        """
        Cập nhật dữ liệu ROI và góc cho camera

        Args:
            camera_id: ID của camera cần cập nhật
            new_data: Dữ liệu ROI mới
            angle: Góc mũi tên mới
        """
        if camera_id not in self.shared_boxes:
            print(f"[ERROR] Camera {camera_id} không tồn tại trong hệ thống")
            return False

        try:
            shared_array = self.shared_boxes[camera_id]
            # Cập nhật mảng dữ liệu chia sẻ
            updated_data = np.array([[p["x"], p["y"]] for p in new_data], dtype=np.float64)
            target_np = np.frombuffer(shared_array.get_obj(), dtype=np.float64).reshape(-1, 2)

            # Đảm bảo kích thước mảng phù hợp
            if len(updated_data) > len(target_np):
                print(f"[WARN] Dữ liệu mới có nhiều điểm hơn ({len(updated_data)} > {len(target_np)})")
                updated_data = updated_data[:len(target_np)]

            for i in range(len(updated_data)):
                target_np[i] = updated_data[i]

            # Cập nhật góc
            shared_angle = self.shared_angles[camera_id]
            shared_angle.value = angle

            return True

        except Exception as e:
            print(f"[ERROR] Không thể cập nhật dữ liệu cho camera {camera_id}: {e}")
            return False

    def update_client_count(self, camera_id, count):
        """
        Cập nhật số lượng client đang xem camera

        Args:
            camera_id: ID của camera
            count: Số lượng client
        """
        if camera_id in self.shared_send_frame:
            try:
                self.shared_send_frame[camera_id].value = count
                return True
            except Exception as e:
                print(f"[ERROR] Không thể cập nhật số lượng client: {e}")
        return False

    def check_processes(self):
        """Thread kiểm tra và khởi động lại các process đã bị dừng"""
        while True:
            try:
                for cam_id, p in list(self.processes.items()):
                    if not p.is_alive():
                        print(f"[WARN] Process cho camera {cam_id} đã dừng. Đang khởi động lại...")
                        if cam_id in self.rtsps:
                            rtsp = self.rtsps[cam_id]
                            # Lấy các thông số hiện tại
                            if cam_id in self.shared_boxes:
                                data = np.array(self.shared_boxes[cam_id]).reshape(-1, 2)
                                points = [{"x": float(p[0]), "y": float(p[1])} for p in data]
                            else:
                                points = []

                            angle = self.shared_angles[cam_id].value if cam_id in self.shared_angles else 0

                            # Xóa camera cũ và khởi động lại
                            self.remove_camera(cam_id)
                            time.sleep(1)
                            self.add_camera(cam_id, rtsp, points, angle)
                        else:
                            print(f"[ERROR] Không thể khởi động lại camera {cam_id}: không tìm thấy URL RTSP")

            except Exception as e:
                print(f"[ERROR] Lỗi trong thread kiểm tra: {e}")

            time.sleep(5)  # Kiểm tra mỗi 5 giây

    def get_cameras(self):
        """Trả về danh sách camera đang hoạt động"""
        return list(self.processes.keys())


# Tạo instance mặc định
ai_plate_service = AIPlateService()
