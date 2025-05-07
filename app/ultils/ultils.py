import math


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
