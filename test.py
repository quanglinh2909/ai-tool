import cv2
import numpy as np

rtsp = "rtsp://admin:Oryza123@192.168.104.2:554/cam/realmonitor?channel=1&subtype=0"
rtsp = (
    f'rtspsrc location={rtsp} latency=0 drop-on-latency=true '
    f'! queue ! rtph264depay ! h264parse ! mppvideodec  !  videorate ! video/x-raw,format=NV12,framerate=10/1 ! appsink'
)
data = [
  {
    "x": 375.1552795031056,
    "y": 64.64876033057851
  },
  {
    "x": 403.9751552795031,
    "y": 67.62396694214877
  },
  {
    "x": 400.99378881987576,
    "y": 76.54958677685951
  },
  {
    "x": 375.1552795031056,
    "y": 79.52479338842976
  }
]
roi_points = np.array([[p["x"], p["y"]] for p in data], np.int32)

cap = cv2.VideoCapture(rtsp)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# tinh lai toado roi
# roi_points[:, 0] = roi_points[:, 0] * w / 640
# roi_points[:, 1] = roi_points[:, 1] * h / 480


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera")
        break
    _frame = frame.copy()
    frame = cv2.resize(frame, (640, 480))
    # Chuyển đổi màu sắc từ BGR sang RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
    frame = cv2.resize(frame, (640, 480))
    # Vẽ hình chữ nhật quanh vùng ROI
    # cv2.polylines(_frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
    # # Vẽ các điểm trong vùng ROI
    # for point in roi_points:
    #     cv2.circle(_frame, tuple(point), 5, (0, 0, 255), -1)
    # Vẽ các điểm trong vùng ROI

    # crop hình chữ nhật quanh vùng ROI
    x, y, w, h = cv2.boundingRect(roi_points)
    roi = frame[y:y+h, x:x+w]
    # roi = cv2.cvtColor(roi, cv2.COLOR_YUV2BGR_NV12)
    cv2.imshow("ROI", roi)

    # Hiển thị frame
    # cv2.imshow("Camera", _frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#     except WebSocketDisconnect: