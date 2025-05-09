import queue
import threading
import time
from collections import defaultdict
from multiprocessing import shared_memory

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from app.ultils.camera_ultil import get_model_plate_platform
from app.ultils.check_platform import get_os_name
from app.ultils.drraw_image import draw_identification_area, draw_direction_vector, draw_moving_path, draw_box, \
    draw_info
from app.ultils.ultils import point_in_polygon, get_direction_vector, direction_similarity


def _send_request( url):
    """Gửi HTTP request trong một thread riêng để không chặn xử lý chính"""
    try:
        requests.get(url, timeout=5)
    except requests.RequestException as e:
        print(f"[ERROR] Không thể gửi request: {e}")

def ai_processor(frame_queue, stop_event, camera_id, shared_array, angle_shared,
                 count_client, shm_name, shape, dtype, ready_event, frame_process_mode=1, is_show=True):
    """
    Luồng xử lý AI với khung hình mới nhất

    Args:
        frame_queue: Queue chứa khung hình mới nhất
        stop_event: Event để dừng luồng
        camera_id: ID của camera
        shared_array: Array chia sẻ cho các điểm ROI
        angle_shared: Giá trị chia sẻ cho góc mũi tên
        count_client: Biến đếm số client đang xem
        shm_name: Tên shared memory
        shape: Kích thước của frame
        dtype: Kiểu dữ liệu của frame
        ready_event: Event báo hiệu frame đã sẵn sàng
        frame_process_mode: Chế độ xử lý frame (1 = chỉ frame mới nhất, >1 = số lượng frame cần xử lý, 0 = tất cả)
        is_show: Hiển thị khung hình hay không
    """
    try:
        platform = get_os_name()
        url_model = get_model_plate_platform(platform)
        model = YOLO(url_model, task="detect")

        # Cấu hình thông số xử lý
        time_wait = None  # Thời gian chờ giữa các lần gửi request
        request_url = "http://192.168.103.97:8090/3"  # URL API để gửi request

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

        max_track_history = 30  # Số lượng tối đa các điểm lịch sử theo dõi
        arrow_similarity_threshold = 0.7  # Ngưỡng tương đồng hướng di chuyển (0.7 ~ 45 độ)

        fps_counter = 0
        fps_timer = time.time()
        ai_fps = 0  # FPS của quá trình xử lý AI

        while not stop_event.is_set():
            try:
                # Quyết định số lượng khung hình cần xử lý
                frames_to_process = []

                if frame_process_mode == 1:
                    # Lấy frame mới nhất (giống với hành vi ban đầu)
                    try:
                        # Trước tiên xóa tất cả frame cũ
                        while not frame_queue.empty():
                            old_frame = frame_queue.get_nowait()
                            if len(frames_to_process) == 0:  # Chỉ giữ frame cuối cùng
                                frames_to_process.append(old_frame)
                    except queue.Empty:
                        pass
                elif frame_process_mode > 1:
                    # Lấy số lượng frame cụ thể (tối đa)
                    for _ in range(frame_process_mode):
                        try:
                            frames_to_process.append(frame_queue.get_nowait())
                            if len(frames_to_process) >= frame_process_mode:
                                break
                        except queue.Empty:
                            break
                else:  # frame_process_mode == 0 hoặc bất kỳ giá trị khác
                    # Xử lý tất cả các frame trong queue
                    try:
                        while not frame_queue.empty():
                            frames_to_process.append(frame_queue.get_nowait())
                    except queue.Empty:
                        pass

                # Nếu không có frame nào để xử lý, đợi và thử lại
                if not frames_to_process:
                    try:
                        frame = frame_queue.get(timeout=1.0)
                        frames_to_process.append(frame)
                    except queue.Empty:
                        time.sleep(0.01)
                        continue

                # Xử lý từng frame trong danh sách
                for frame in frames_to_process:
                    # Tạo bản sao để hiển thị
                    display_frame = frame.copy()

                    # Đọc dữ liệu từ shared memory
                    data = np.array(shared_array).reshape(-1, 2)
                    angle = angle_shared.value
                    client_count = count_client.value

                    # Vẽ vùng nhận diện
                    roi_points_drawn, roi_points = draw_identification_area(data, display_frame, is_draw=True)

                    # Vẽ mũi tên chỉ hướng
                    arrow_vector = draw_direction_vector(roi_points_drawn, display_frame, angle, is_draw=True)

                    # Đếm FPS
                    fps_counter += 1
                    if time.time() - fps_timer > 1.0:
                        ai_fps = fps_counter
                        fps_counter = 0
                        fps_timer = time.time()

                    # Nhận diện và theo dõi đối tượng
                    start_time = time.time()
                    result = model.track(frame, persist=True, verbose=False)[0]
                    inference_time = time.time() - start_time

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
                            if int(class_id) != 5:  # Nếu muốn chỉ theo dõi một loại đối tượng cụ thể
                                continue

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
                                            threading.Thread(
                                                target=_send_request,
                                                args=(request_url,),
                                                daemon=True
                                            ).start()
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
                            # Dọn dẹp các track_history không còn được sử dụng
                            if tid in track_history:
                                del track_history[tid]

                    # Hiển thị thông tin tổng hợp
                    draw_info(frame, display_frame, objects_following_arrow, objects_in_roi, is_draw_display=True,
                              is_draw_frame=True)

                    # Hiển thị thông tin FPS và thời gian xử lý
                    cv2.putText(display_frame, f"AI FPS: {ai_fps}, Inference: {inference_time:.3f}s",
                                (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                    # Hiển thị chế độ xử lý frame
                    mode_text = "Che do: "
                    if frame_process_mode == 1:
                        mode_text += "Frame moi nhat"
                    elif frame_process_mode > 1:
                        mode_text += f"{frame_process_mode} frames"
                    else:
                        mode_text += "Tat ca frames"
                    cv2.putText(display_frame, mode_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

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
                            stop_event.set()
                            break

            except Exception as e:
                print(f"[ERROR] Lỗi trong vòng lặp xử lý AI: {e}")
                time.sleep(0.1)  # Ngăn ngừa vòng lặp vô hạn khi có lỗi

    except Exception as e:
        print(f"[ERROR] Lỗi trong ai_processor: {e}")
    finally:
        # Đảm bảo giải phóng tài nguyên khi processor kết thúc
        try:
            if 'existing_shm' in locals():
                existing_shm.close()
            if is_show and 'cv2' in locals():
                cv2.destroyAllWindows()
        except Exception as cleanup_error:
            print(f"[ERROR] Lỗi khi dọn dẹp: {cleanup_error}")
