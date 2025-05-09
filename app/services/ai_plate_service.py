import multiprocessing
from collections import defaultdict
from multiprocessing import shared_memory

import cv2
import time
import threading
import numpy as np
import requests
from ultralytics import YOLO

from app.constants.platform_enum import PlatformEnum
from app.ultils.camera_ultil import get_rtsp_platform, get_model_plate_platform
from app.ultils.check_platform import get_os_name
from app.ultils.drraw_image import draw_identification_area, draw_direction_vector, draw_moving_path, draw_box, \
    draw_info
from app.ultils.ultils import point_in_polygon, get_direction_vector, direction_similarity, calculate_arrow_end


class AIPlateService:
    def __init__(self):
        self.processes = {}
        self.rtsps = {}
        self.shared_boxes = {}
        self.shared_memories = {}
        self.shared_angles = {}
        self.shared_send_frame = {}
        # Khởi động thread giám sát quy trình
        threading.Thread(target=self.check_processes, daemon=True).start()

    def worker(self, camera_id, rtsp, shared_array, angle_shared, count_client, shm_name, shape, dtype, ready_event,
               is_show=True):
        try:
            platform = get_os_name()
            url_model = get_model_plate_platform(platform)
            model = YOLO(url_model, task="detect")

            """
                 - nếu time_wait = None thì cho phép gửi yêu cầu đúng 1 lần
                 - nếu time_wait != None thì cho phép gửi yêu cầu nhiều lần nếu thời gian chênh lệch >= time_wait
            """
            time_wait = None  # Có thể điều chỉnh nếu cần thiết cho phép gửi request nhiều lần
            request_url = "http://192.168.103.97:8090/3"  # URL cho requests - đặt ở biến để dễ thay đổi

            # Theo dõi thời gian gửi request cho mỗi đối tượng
            object_timers = {}  # {track_id: last_event_time}

            try:
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                frame_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
            except Exception as e:
                print(f"[ERROR] Shared memory error: {e}")
                return

            objects_in_roi = {}  # Các đối tượng đang ở trong vùng ROI
            objects_following_arrow = {}  # Các đối tượng đang di chuyển theo mũi tên
            track_history = defaultdict(list)  # Lịch sử di chuyển của các đối tượng

            # Chuẩn bị URL RTSP
            rtsp = get_rtsp_platform(rtsp, platform)

            reconnect_delay = 5  # Số giây chờ trước khi kết nối lại
            max_track_history = 30  # Số lượng tối đa các điểm lịch sử theo dõi
            arrow_similarity_threshold = 0.7  # Ngưỡng tương đồng hướng di chuyển (0.7 ~ 45 độ)

            while True:
                print(f"[INFO] Đang kết nối tới: {rtsp}")
                cap = cv2.VideoCapture(rtsp)
                if not cap.isOpened():
                    print(f"[WARN] Không thể kết nối. Thử lại sau {reconnect_delay} giây...")
                    time.sleep(reconnect_delay)
                    continue

                print("[INFO] Đã kết nối với camera.")
                frame_count = 0

                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            print("[WARN] Mất frame. Đang kết nối lại...")
                            break  # Thoát khỏi vòng lặp đọc để reconnect

                        frame_count += 1
                        frame = cv2.resize(frame, (640, 480))

                        # Chuyển đổi màu sắc từ BGR sang RGB nếu cần
                        if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
                            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

                        display_frame = frame.copy()

                        # Đọc dữ liệu từ shared memory
                        data = np.array(shared_array).reshape(-1, 2)
                        angle = angle_shared.value
                        client_count = count_client.value

                        # Vẽ vùng nhận diện
                        roi_points_drawn, roi_points = draw_identification_area(data, display_frame, is_draw=True)

                        # Vẽ mũi tên chỉ hướng
                        arrow_vector = draw_direction_vector(roi_points_drawn, display_frame, angle, is_draw=True)

                        # Nhận diện và theo dõi đối tượng
                        result = model.track(frame, persist=True, verbose=False)[0]

                        # Xử lý kết quả nhận diện nếu có
                        current_objects = set()

                        if result.boxes and result.boxes.id is not None:
                            boxes = result.boxes.xywh.cpu()
                            track_ids = result.boxes.id.int().cpu().tolist()
                            class_ids = result.boxes.cls.cpu().tolist()

                            # Kiểm tra từng đối tượng
                            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                                x, y, w, h = box
                                current_objects.add(track_id)

                                # Lọc theo class_id nếu cần
                                # if int(class_id) != 5:  # Nếu muốn chỉ theo dõi một loại đối tượng cụ thể
                                #     continue

                                center_point = (float(x), float(y))

                                # Lưu vết di chuyển
                                track = track_history[track_id]
                                track.append(center_point)
                                if len(track) > max_track_history:
                                    track.pop(0)

                                # Vẽ đường di chuyển
                                draw_moving_path(frame, display_frame, track, is_draw_display=True, is_draw_frame=True)

                                # Vẽ box cho đối tượng
                                draw_box(frame, display_frame, x, y, w, h, is_draw_display=True, is_draw_frame=True)

                                in_roi = False
                                if roi_points is not None:
                                    # Kiểm tra xem đối tượng có nằm trong ROI không
                                    in_roi = point_in_polygon(center_point, roi_points)

                                # Nếu đối tượng nằm trong ROI
                                if in_roi:
                                    objects_in_roi[track_id] = True

                                    # Xác định vector di chuyển khi có ít nhất 2 điểm
                                    if len(track) > 1:
                                        movement_vector = get_direction_vector(track)
                                        # Tính độ tương đồng giữa hướng di chuyển và hướng mũi tên
                                        similarity = direction_similarity(movement_vector, arrow_vector)

                                        # Kiểm tra đối tượng có di chuyển theo hướng mũi tên không
                                        following_arrow = similarity > arrow_similarity_threshold

                                        # Lưu trạng thái
                                        objects_following_arrow[track_id] = following_arrow

                                        # Xử lý trạng thái và hiển thị
                                        current_time = time.time()

                                        if following_arrow:
                                            status = "Theo huong"
                                            color = (0, 255, 0)  # Xanh lá

                                            # Logic gửi request cho từng đối tượng
                                            should_send_request = False

                                            # Trường hợp 1: Chưa từng gửi request cho object này
                                            if track_id not in object_timers:
                                                should_send_request = True
                                                object_timers[track_id] = current_time
                                            # Trường hợp 2: Đã gửi request trước đó và đủ thời gian chờ
                                            elif time_wait is not None and (
                                                    current_time - object_timers[track_id] > time_wait):
                                                should_send_request = True
                                                object_timers[track_id] = current_time

                                            # Gửi request nếu thỏa điều kiện
                                            if should_send_request:
                                                print(f"Theo huong, ID: {track_id}")
                                                try:
                                                    requests.get(request_url, timeout=1)
                                                except requests.RequestException as e:
                                                    print(f"[ERROR] Không thể gửi request: {e}")
                                        else:
                                            status = "Khong theo huong"
                                            color = (0, 0, 255)  # Đỏ

                                        # Hiển thị nhãn
                                        label = f"ID: {track_id}, {status}"
                                        cv2.putText(display_frame, label, (int(x), int(y) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                elif track_id in objects_in_roi:
                                    # Đối tượng đã rời khỏi ROI
                                    print(f"Đối tượng đã rời khỏi ROI: {track_id}")
                                    del objects_in_roi[track_id]
                                    if track_id in objects_following_arrow:
                                        del objects_following_arrow[track_id]

                                    # Xóa timer khi đối tượng rời khỏi ROI
                                    if track_id in object_timers:
                                        del object_timers[track_id]

                            # Dọn dẹp các đối tượng không còn được theo dõi
                            ids_to_remove = [tid for tid in object_timers if tid not in current_objects]
                            for tid in ids_to_remove:
                                del object_timers[tid]

                        # Hiển thị thông tin tổng hợp
                        draw_info(frame, display_frame, objects_following_arrow, objects_in_roi, is_draw_display=True,
                                  is_draw_frame=True)

                        # Cập nhật frame vào shared memory nếu có client đang xem
                        if client_count > 0:
                            frame = cv2.resize(frame, (640, 480))
                            np.copyto(frame_np, frame)
                            # Đánh dấu là đã sẵn sàng
                            ready_event.set()

                        # Hiển thị frame nếu cần
                        if is_show:
                            cv2.imshow(f'Camera: {camera_id}', display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("[INFO] Nhận tín hiệu thoát. Đóng stream.")
                                cap.release()
                                cv2.destroyAllWindows()
                                return
                        else:
                            # Nếu không cần hiển thị, chỉ đợi một khoảng thời gian ngắn
                            time.sleep(0.01)

                    except Exception as e:
                        print(f"[ERROR] Lỗi trong vòng lặp xử lý: {e}")
                        time.sleep(0.1)  # Ngăn ngừa vòng lặp vô hạn khi có lỗi

                # Dọn dẹp tài nguyên khi kết thúc vòng lặp
                cap.release()
                if is_show:
                    cv2.destroyAllWindows()
                print(f"[INFO] Kết nối lại sau {reconnect_delay} giây...")
                time.sleep(reconnect_delay)

        except Exception as e:
            print(f"[ERROR] Lỗi trong worker: {e}")
        finally:
            # Đảm bảo giải phóng tài nguyên khi worker kết thúc
            try:
                if 'existing_shm' in locals():
                    existing_shm.close()
                if 'cap' in locals() and cap is not None:
                    cap.release()
                if is_show and 'cv2' in locals():
                    cv2.destroyAllWindows()
            except Exception as cleanup_error:
                print(f"[ERROR] Lỗi khi dọn dẹp: {cleanup_error}")

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

                # Khởi động worker process
                process = multiprocessing.Process(
                    target=self.worker,
                    args=(camera_id, rtsp, shared_array, angle_shared, count_client,
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
                    "ready_event": ready_event
                }
                self.shared_angles[camera_id] = angle_shared
                self.shared_send_frame[camera_id] = count_client

                print(f"Đã thêm camera {camera_id} thành công")
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