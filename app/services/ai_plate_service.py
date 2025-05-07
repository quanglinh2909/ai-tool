import multiprocessing
from collections import defaultdict
from multiprocessing import shared_memory

import cv2
import time
import threading
import numpy as np
import requests
from ultralytics import YOLO

from app.ultils.ultils import point_in_polygon, get_direction_vector, direction_similarity, calculate_arrow_end


class AIPlateService:
    def __init__(self):
        self.processes = {}
        self.rtsps = {}
        self.shared_boxes = {}
        self.shared_memories = {}
        self.shared_angles = {}
        threading.Thread(target=self.check_processes, daemon=True).start()

    def worker(self, camera_id, rtsp,shared_array ,angle_shared,shm_name, shape, dtype, ready_event):
        model = YOLO("/home/orangepi/vehicle_plate_2_rknn_model")
        id_current = None
        time_event = time.time()
        time_wait = None
        shared_array_current = None
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        frame_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        roi_points = None
        arrow_start = None
        arrow_end = None
        objects_in_roi = {}
        objects_following_arrow = {}
        angle_current = 0
        track_history = defaultdict(lambda: [])
        arrow_vector = (0, 0)

        pipeline = (
            f"rtspsrc location={rtsp} latency=0 drop-on-latency=true ! queue ! rtph264depay ! h264parse ! mppvideodec  !  videorate ! video/x-raw,format=NV12,framerate=15/1 ! "
            f"appsink drop=true sync=false"
        )

        while True:
            print(f"[INFO] Trying to connect to: {rtsp}")
            cap = cv2.VideoCapture(pipeline)
            if not cap.isOpened():
                print("[WARN] Cannot connect. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            print("[INFO] Connected to camera.")
            while cap.isOpened():
                data = np.array(shared_array).reshape(-1, 2)
                angle = angle_shared.value
                if shared_array_current is None or  np.array_equal(shared_array_current , data) is False or angle_current != angle:
                    shared_array_current = data
                    angle_current = angle
                    # Chuyển đổi dữ liệu thành mảng numpy
                    data_box = []
                    for i in range(len(data)):
                        x = data[i][0]
                        y = data[i][1]
                        if x == -1 and y == -1:
                            continue
                        data_box.append({"x": x, "y": y})
                    # Store the track history


                    if len(data_box) >=3   :
                        roi_points = np.array([[p["x"], p["y"]] for p in data_box], np.int32)
                        # Định nghĩa mũi tên chỉ hướng (điểm bắt đầu và điểm kết thúc)
                        arrow_start = np.mean(roi_points, axis=0).astype(int)
                        arrow_length = 100  # Có thể điều chỉnh độ dài mũi tên

                        arrow_end = calculate_arrow_end(arrow_start, angle, arrow_length)
                        # Tính vector hướng của mũi tên (sẽ dùng để so sánh với hướng di chuyển của đối tượng)
                        arrow_vector = (arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1])
                        roi_points = roi_points.reshape((-1, 1, 2))
                        # Lưu trạng thái của các đối tượng trong ROI


                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Lost frame. Reconnecting...")
                    break  # Thoát khỏi vòng lặp đọc để reconnect
                frame = cv2.resize(frame, (640, 480))
                # Chuyển đổi màu sắc từ BGR sang RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                # frame = cv2.resize(frame, (640, 480))


                # Tạo bản sao của frame để tránh vẽ đè lên
                display_frame = frame.copy()
                if roi_points is not None:
                    # Vẽ vùng ROI
                    cv2.polylines(display_frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
                if arrow_start is not None and arrow_end is not None:
                    # Vẽ mũi tên chỉ hướng
                    cv2.arrowedLine(display_frame, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)
                # Run YOLO tracking on the frame, persisting tracks between frames
                result = model.track(frame, persist=True, verbose=False)[0]

                if result.boxes and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    class_ids = result.boxes.cls.cpu().tolist()

                    # Kiểm tra từng đối tượng
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        x, y, w, h = box
                        if int(class_id) != 5:
                            continue
                        center_point = (float(x), float(y))

                        # Lưu vết
                        track = track_history[track_id]
                        track.append(center_point)
                        if len(track) > 30:  # retain 30 tracks for 30 frames
                            track.pop(0)

                        # Vẽ đường di chuyển
                        if len(track) > 1:
                            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(display_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                        # Vẽ box cho đối tượng
                        x1, y1 = int(x - w / 2), int(y - h / 2)
                        x2, y2 = int(x + w / 2), int(y + h / 2)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        in_roi = False
                        if roi_points is not None:
                            # Kiểm tra xem đối tượng có nằm trong ROI không
                            in_roi = point_in_polygon(center_point, roi_points)

                        # Nếu đối tượng trong ROI, kiểm tra hướng di chuyển
                        if in_roi:
                            objects_in_roi[track_id] = True

                            # Xác định vector di chuyển của đối tượng
                            if len(track) > 1:
                                movement_vector = get_direction_vector(track)
                                # Tính độ tương đồng giữa hướng di chuyển và hướng mũi tên
                                similarity = direction_similarity(movement_vector, arrow_vector)

                                # Nếu đối tượng di chuyển theo hướng gần với mũi tên (cosine > 0.7 ~ 45 độ)
                                following_arrow = similarity > 0.7

                                # Lưu trạng thái và hiển thị
                                objects_following_arrow[track_id] = following_arrow

                                # Hiển thị thông tin trên frame
                                if following_arrow:
                                    status = "Theo huong "
                                    color = (0, 255, 0)  # Xanh lá
                                    if id_current != track_id or (
                                            time_wait is not None and time.time() - time_event > time_wait  ):
                                        id_current = track_id
                                        time_event = time.time()
                                        print("Theo huong", track_id)
                                        requests.get("http://192.168.103.97:8090/3")
                                else:
                                    status = "Khong theo huong"
                                    color = (0, 0, 255)  # Đỏ

                                label = f"ID: {track_id}, {status}"
                                cv2.putText(display_frame, label, (int(x), int(y) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                # Vẽ vector hướng di chuyển của đối tượng (thu nhỏ để dễ nhìn)
                                movement_scale = 30
                                movement_end = (int(x + movement_vector[0] / movement_scale),
                                                int(y + movement_vector[1] / movement_scale))
                                cv2.arrowedLine(display_frame, (int(x), int(y)), movement_end, color, 2)

                        elif track_id in objects_in_roi:
                            # Đối tượng đã rời khỏi ROI
                            print("Đối tượng đã rời khỏi ROI", track_id)
                            del objects_in_roi[track_id]
                            if track_id in objects_following_arrow:
                                del objects_following_arrow[track_id]

                            if track_id == id_current:
                                id_current = None

                if roi_points is not None:
                # Hiển thị thông tin tổng hợp
                    following_count = sum(1 for value in objects_following_arrow.values() if value)
                    cv2.putText(display_frame, f"Trong vùng: {len(objects_in_roi)}", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Trong vùng: {len(objects_in_roi)}", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if arrow_start is not None and arrow_end is not None:
                    cv2.putText(display_frame, f"Theo hướng mũi tên: {following_count}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.putText(frame, f"Theo hướng mũi tên: {following_count}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # Display the annotated frame - chỉ gọi imshow một lần mỗi vòng lặp

                np.copyto(frame_np, frame)
                # # Đánh dấu là đã sẵn sàng
                ready_event.set()

                cv2.imshow(f'Camera: {camera_id}', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quit signal received. Closing stream.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Reconnecting in 5 seconds...")
            time.sleep(5)

    def add_camera(self, camera_id, rtsp):
        if camera_id in self.processes:
            print(f"Camera {camera_id} already running.")
            return
        data_box = []
        for i in range(1, 100):
            data_box.append({"x": -1, "y": -1})


        # Chuyển đổi dữ liệu thành mảng numpy
        shared_data = np.array([[p["x"], p["y"]] for p in data_box], dtype=np.float64)
        shared_array = multiprocessing.Array('d', shared_data.flatten())  # 'd' là kiểu float64
        shape = (320,640, 3)  # height, width, channels
        dtype = np.uint8
        size = np.prod(shape)  # 640 * 480 * 3 = 921600
        shm_global = shared_memory.SharedMemory(create=True, size=size)
        ready_event = multiprocessing.Event()
        # Tạo Array chia sẻ

        angle = multiprocessing.Value('i', 0)


        process = multiprocessing.Process(target=self.worker, args=(camera_id, rtsp,shared_array,angle,shm_global.name, shape, dtype, ready_event), daemon=True)
        process.start()


        self.processes[camera_id] = process
        self.rtsps[camera_id] = rtsp
        self.shared_boxes[camera_id] = shared_array
        self.shared_memories[camera_id] = {"shm": shm_global, "shape": shape, "dtype": dtype,"ready_event":ready_event}
        self.shared_angles[camera_id] = angle
        print(f"add_camera {camera_id}")

    def remove_camera(self, id_camera):
        if id_camera in self.processes:
            self.processes[id_camera].terminate()
            self.processes[id_camera].join()
            del self.processes[id_camera]
            print(f"Camera {id_camera} stopped.")

        if id_camera in self.rtsps:
            del self.rtsps[id_camera]
            print(f"Camera {id_camera} removed from list.")

    def update_shared_array(self, camera_id, new_data,angle):
        shared_array = self.shared_boxes[camera_id]
        # Update the shared data array
        updated_data = np.array([[p["x"], p["y"]] for p in new_data], dtype=np.float64)
        target_np = np.frombuffer(shared_array.get_obj(), dtype=np.float64).reshape(-1, 2)
        for i in range(len(updated_data)):
            target_np[i] = updated_data[i]



        shared_angle = self.shared_angles[camera_id]
        shared_angle.value = angle


    def check_processes(self):
        while True:
            for cam_id, p in list(self.processes.items()):
                if not p.is_alive():
                    print("Worker process died. Restarting...")
                    rtsp = self.rtsps[cam_id]
                    self.remove_camera(cam_id)
                    time.sleep(1)
                    self.add_camera(cam_id, rtsp)
            time.sleep(2)

ai_plate_service = AIPlateService()
# if __name__ == '__main__':
#
#     # ai_plate_service.add_camera(1, "rtsp://admin:Oryza123@192.168.104.2:554/cam/realmonitor?channel=1&subtype=0")
#     ai_plate_service.add_camera(1, "/home/linh/Documents/ai-tool/test/output_vao.avi")
#     # ai_plate_service.add_camera(2, "rtsp://admin:Oryza123@192.168.104.2:554/cam/realmonitor?channel=1&subtype=0")
#
#     ai_plate_service.check_processes()
