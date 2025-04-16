from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import math

# Load the YOLO model
model = YOLO("vehicle_plate_2.pt")

# Open the video file
video_path = "output_vao.avi"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])


# Định nghĩa vùng ROI xung quanh mũi tên
data = [
  {
    "x": 41.73913043478261,
    "y": 426.6322314049587
  },
  {
    "x": 290.18633540372673,
    "y": 160.8471074380165
  },
  {
    "x": 449.1925465838509,
    "y": 83.49173553719007
  },
  {
    "x": 616.1490683229814,
    "y": 124.15289256198346
  },
  {
    "x": 595.27950310559,
    "y": 456.38429752066116
  }
]
roi_points = np.array([[p["x"], p["y"]] for p in data], np.int32)
# Định nghĩa mũi tên chỉ hướng (điểm bắt đầu và điểm kết thúc)
arrow_start = np.mean(roi_points, axis=0).astype(int)
arrow_length = 100  # Có thể điều chỉnh độ dài mũi tên


def calculate_arrow_end(start_point, angle_degrees, length):
    # Chuyển đổi góc từ độ sang radian
    angle_radians = math.radians(angle_degrees)

    # Tính toạ độ điểm cuối dựa trên góc
    # Lưu ý: Trong hệ toạ độ y-xuống (như trong OpenCV), góc 0 độ là hướng sang phải,
    # và góc tăng theo chiều kim đồng hồ
    dx = length * math.cos(angle_radians)
    dy = length * math.sin(angle_radians)

    # Tính điểm cuối
    end_x = int(start_point[0] + dx)
    end_y = int(start_point[1] + dy)

    return (end_x, end_y)

angle = -45  # Có thể thay đổi từ 0 đến 360 độ
arrow_end = calculate_arrow_end(arrow_start, angle, arrow_length)

# Tính vector hướng của mũi tên (sẽ dùng để so sánh với hướng di chuyển của đối tượng)
arrow_vector = (arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1])
arrow_length = math.sqrt(arrow_vector[0] ** 2 + arrow_vector[1] ** 2)
arrow_unit_vector = (arrow_vector[0] / arrow_length, arrow_vector[1] / arrow_length)

roi_points = roi_points.reshape((-1, 1, 2))
# Lưu trạng thái của các đối tượng trong ROI
objects_in_roi = {}
objects_following_arrow = {}

# # Đóng tất cả cửa sổ đang mở trước khi bắt đầu
# cv2.destroyAllWindows()

# Tạo cửa sổ với tên cố định
window_name = "Tracking theo hướng mũi tên"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


# Hàm kiểm tra điểm có nằm trong đa giác
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0][0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n][0]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# Hàm tính độ tương đồng giữa hai vector (cosine similarity)
def direction_similarity(vec1, vec2):
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    cosine = dot_product / (magnitude1 * magnitude2)
    return cosine


# Hàm xác định vector hướng từ lịch sử di chuyển
def get_direction_vector(track):
    if len(track) < 2:
        return (0, 0)

    # Lấy vị trí đầu và cuối để xác định hướng tổng thể
    x1, y1 = track[0]
    x2, y2 = track[-1]

    return (x2 - x1, y2 - y1)


try:
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            print("Kết thúc video")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize the frame

        # Tạo bản sao của frame để tránh vẽ đè lên
        display_frame = frame.copy()

        # Vẽ vùng ROI
        cv2.polylines(display_frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Vẽ mũi tên chỉ hướng
        cv2.arrowedLine(display_frame, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)

        # Run YOLO tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True, verbose=False)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            # Kiểm tra từng đối tượng
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x, y, w, h = box
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

                # Vẽ box cho đối tượng
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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

        # Hiển thị thông tin tổng hợp
        following_count = sum(1 for value in objects_following_arrow.values() if value)
        cv2.putText(display_frame, f"Trong vùng: {len(objects_in_roi)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Theo hướng mũi tên: {following_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the annotated frame - chỉ gọi imshow một lần mỗi vòng lặp
        cv2.imshow(window_name, display_frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q hoặc phím ESC
            print("Đã nhấn phím thoát")
            break

except Exception as e:
    print(f"Lỗi: {str(e)}")
finally:
    # Đảm bảo giải phóng tài nguyên dù có lỗi xảy ra
    print("Đang đóng ứng dụng...")
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng ứng dụng")