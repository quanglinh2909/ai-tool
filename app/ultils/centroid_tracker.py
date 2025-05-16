import numpy as np
from scipy.spatial import distance
import cv2
import time
from datetime import datetime, timedelta

# Thêm vào đầu file centroid_tracker.py
# filepath: d:\Python\ai-tool\test\centroid_tracker.py
# Định nghĩa lại các class hoặc import từ nơi khác


class CentroidTracker:
    """
    Centroid Tracker - Thuật toán tracking nhẹ nhất
    Chỉ sử dụng khoảng cách Euclid giữa các tâm bbox
    """

    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # format {ID: centroid}
        self.disappeared = {}  # format {ID: count}
        self.bbox = {}  # format {ID: bbox}
        self.class_ids = {}  # format {ID: class_id}
        self.scores = {}  # format {ID: score}
        self.colors = {}  # format {ID: color}

        # Thêm thông tin về thời gian tracking
        self.appear_time = {}  # format {ID: timestamp xuất hiện}
        self.track_duration = {}  # format {ID: thời lượng theo dõi}
        self.last_update_time = {}  # format {ID: timestamp cập nhật gần nhất}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _get_centroid(self, bbox):
        """
        Tính tọa độ tâm của bbox
        """
        x1, y1, x2, y2 = bbox
        cX = int((x1 + x2) / 2)
        cY = int((y1 + y2) / 2)
        return (cX, cY)

    def update(self, boxes, scores=None, classes=None):
        """
        Cập nhật tracker với bbox mới
        """
        # Lưu danh sách ID hiện tại để xác định các đối tượng đã biến mất
        current_ids = set(self.objects.keys())
        new_ids = set()

        # Trường hợp không có bbox nào
        if len(boxes) == 0:
            # Đánh dấu tất cả objects hiện tại là disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    # Thông báo đối tượng mất tracking
                    self._report_object_lost(object_id,classes)
                    self.deregister(object_id)

            return self._get_result()

        # Tính tâm của tất cả bbox mới
        centroids = [self._get_centroid(box) for box in boxes]

        # Nếu chưa có object nào được theo dõi
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                object_id = self.register(centroids[i], boxes[i],
                                          scores[i] if scores is not None else None,
                                          classes[i] if classes is not None else None,classes)
                new_ids.add(object_id)
        else:
            # Lấy ID và tâm của các object đang theo dõi
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Tính ma trận khoảng cách giữa các centroids
            D = distance.cdist(object_centroids, centroids)

            # Tìm chỉ số có khoảng cách nhỏ nhất cho mỗi hàng
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Đánh dấu những hàng và cột đã xử lý để tránh duplicate
            used_rows = set()
            used_cols = set()

            # Duyệt qua các (row, col) combination
            for (row, col) in zip(rows, cols):
                # Bỏ qua nếu đã xử lý hoặc khoảng cách quá lớn
                if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                    continue

                # Lấy ID của object hiện tại và reset disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.bbox[object_id] = boxes[col]
                if scores is not None:
                    self.scores[object_id] = scores[col]
                if classes is not None:
                    self.class_ids[object_id] = classes[col]
                self.disappeared[object_id] = 0

                # Cập nhật thời gian tracking
                current_time = time.time()
                self.last_update_time[object_id] = current_time
                track_duration = current_time - self.appear_time[object_id]
                self.track_duration[object_id] = track_duration

                # Đánh dấu đã xử lý
                used_rows.add(row)
                used_cols.add(col)
                new_ids.add(object_id)

            # Tìm các hàng và cột chưa xử lý
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            # Nếu có objects biến mất
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        # Thông báo đối tượng mất tracking
                        self._report_object_lost(object_id,classes)
                        self.deregister(object_id)

            # Nếu có objects mới xuất hiện
            else:
                for col in unused_cols:
                    object_id = self.register(centroids[col], boxes[col],
                                              scores[col] if scores is not None else None,
                                              classes[col] if classes is not None else None,classes)
                    new_ids.add(object_id)

        # Xác định đối tượng nào đã biến mất (không còn trong new_ids)
        disappeared_ids = current_ids - new_ids
        for object_id in disappeared_ids:
            if object_id in self.objects:  # Kiểm tra xem đối tượng còn tồn tại không
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    # Thông báo đối tượng mất tracking
                    self._report_object_lost(object_id,classes)
                    self.deregister(object_id)

        return self._get_result()

    def register(self, centroid, bbox, score=None, class_id=None,classes=None):
        """
        Đăng ký object mới
        """
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bbox[object_id] = bbox
        self.disappeared[object_id] = 0
        if score is not None:
            self.scores[object_id] = score
        if class_id is not None:
            self.class_ids[object_id] = class_id

        # Tạo màu ngẫu nhiên cho object
        self.colors[object_id] = tuple(np.random.randint(0, 255, size=3).tolist())

        # Thêm thông tin về thời gian
        current_time = time.time()
        self.appear_time[object_id] = current_time
        self.last_update_time[object_id] = current_time
        self.track_duration[object_id] = 0

        # Thông báo đối tượng mới
        self._report_new_object(object_id,classes)

        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        """
        Hủy đăng ký object khi không còn theo dõi
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bbox[object_id]
        if object_id in self.scores:
            del self.scores[object_id]
        if object_id in self.class_ids:
            del self.class_ids[object_id]
        if object_id in self.colors:
            del self.colors[object_id]

        # Xóa thông tin thời gian
        if object_id in self.appear_time:
            del self.appear_time[object_id]
        if object_id in self.last_update_time:
            del self.last_update_time[object_id]
        if object_id in self.track_duration:
            del self.track_duration[object_id]

    def _report_new_object(self, object_id,class_names):
        """
        Thông báo đối tượng mới xuất hiện
        """
        class_id = self.class_ids.get(object_id)
        class_name = class_names[class_id] if class_id is not None and class_id < len(class_names) else "Unknown"
        score = self.scores.get(object_id, 0)

        current_time = datetime.now().strftime("%H:%M:%S")
        # print(f"[{current_time}] ID:{object_id} {class_name} (score: {score:.2f}) XUẤT HIỆN")

    def _report_object_lost(self, object_id,class_names):
        """
        Thông báo đối tượng mất tracking
        """
        class_id = self.class_ids.get(object_id)
        class_name = class_names[class_id] if class_id is not None and class_id < len(class_names) else "Unknown"

        # Tính thời gian theo dõi
        duration = self.track_duration.get(object_id, 0)
        duration_str = str(timedelta(seconds=int(duration)))

        current_time = datetime.now().strftime("%H:%M:%S")
        # print(f"[{current_time}] ID:{object_id} {class_name} MẤT TRACKING sau {duration_str}")

    def _get_result(self):
        """
        Trả về kết quả tracking
        """
        result = []
        for object_id in self.objects:
            bbox = self.bbox[object_id]
            score = self.scores.get(object_id, None)
            class_id = self.class_ids.get(object_id, None)
            color = self.colors.get(object_id, (0, 255, 0))

            # Thêm thời gian tracking vào kết quả
            track_duration = self.track_duration.get(object_id, 0)

            result.append((bbox, object_id, score, class_id, color, track_duration))

        return result


def draw_tracks(image, tracks, class_names=None):
    """
    Vẽ các track lên ảnh và hiển thị thời gian tracking
    """
    for track in tracks:
        bbox, track_id, score, class_id, color, duration = track
        x1, y1, x2, y2 = [int(i) for i in bbox]

        # Vẽ bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Vẽ centroid
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.circle(image, centroid, 4, color, -1)

        # Hiển thị ID và class
        class_name = class_names[class_id] if class_id is not None and class_names is not None else "Unknown"
        score_text = f":{score:.2f}" if score is not None else ""

        # Định dạng thời gian tracking
        duration_str = str(timedelta(seconds=int(duration)))
        if "day" in duration_str:
            # Định dạng ngắn gọn hơn cho thời gian dài
            days, time = duration_str.split(", ")
            hours, minutes, seconds = time.split(":")
            duration_str = f"{days}, {hours}h{minutes}m"
        else:
            hours, minutes, seconds = duration_str.split(":")
            if int(hours) > 0:
                duration_str = f"{hours}h{minutes}m{seconds}s"
            elif int(minutes) > 0:
                duration_str = f"{minutes}m{seconds}s"
            else:
                duration_str = f"{seconds}s"

        label = f"ID:{track_id} {class_name}{score_text}"
        time_label = f"Time: {duration_str}"

        # Vẽ nền cho text
        cv2.rectangle(image, (x1, y1 - 40), (x1 + max(len(label), len(time_label)) * 8, y1), color, -1)

        # Hiển thị text
        cv2.putText(image, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, time_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image