import cv2
import numpy as np

from app.ultils.ultils import calculate_arrow_end


def draw_identification_area(data, display_frame, is_draw=True):
    data_box = []
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        if x == -1 and y == -1:
            continue
        data_box.append({"x": x, "y": y})
    # Store the track history

    roi_points_drawn = np.array([[p["x"], p["y"]] for p in data_box], np.int32)
    roi_points = roi_points_drawn.reshape((-1, 1, 2))

    if is_draw:
        # Vẽ đường viền đa giác
        cv2.polylines(display_frame, [roi_points_drawn], isClosed=True, color=(0, 255, 0), thickness=2)
        # Vẽ điểm rõ nét tại từng điểm (x, y)
        for p in roi_points:
            cv2.circle(display_frame, (p[0][0], p[0][1]), 5, (255, 0, 0), -1)
    return roi_points_drawn, roi_points


def draw_direction_vector(roi_points, display_frame, angle, is_draw=True):
    arrow_start = np.mean(roi_points, axis=0).astype(int)
    arrow_length = 100  # Có thể điều chỉnh độ dài mũi tên
    arrow_end = calculate_arrow_end(arrow_start, angle, arrow_length)
    if is_draw:
        cv2.arrowedLine(display_frame, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)
    arrow_vector = (arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1])
    return arrow_vector


# Vẽ đường di chuyển
def draw_moving_path(frame, display_frame, track, is_draw_display=True, is_draw_frame=True):
    if len(track) > 1:
        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
        if is_draw_display:
            cv2.polylines(display_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        if is_draw_frame:
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

# Vẽ box cho đối tượng
def draw_box(frame, display_frame, x, y, w, h, is_draw_display=True, is_draw_frame=True):
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    if is_draw_display:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if is_draw_frame:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Hiển thị thông tin tổng hợp
def draw_info(frame,display_frame, objects_following_arrow,objects_in_roi, is_draw_display=True,is_draw_frame=True):
    following_count = sum(1 for value in objects_following_arrow.values() if value)
    if is_draw_display:
        cv2.putText(display_frame, f"Trong vung: {len(objects_in_roi)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.putText(display_frame, f"Theo huong mui ten: {following_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if is_draw_frame:
        cv2.putText(frame, f"Trong vung: {len(objects_in_roi)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Theo huong mui ten: {following_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)