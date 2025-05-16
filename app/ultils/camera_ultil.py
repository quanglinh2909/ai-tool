import queue
import time

import av

from app.constants.platform_enum import PlatformEnum
from app.ultils.check_platform import get_os_name


def get_rtsp_platform(rtsp: str, platform=None) -> str:
    if platform is None:
        platform = get_os_name()
    if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
        pipeline = (
            f"rtspsrc location={rtsp} latency=0 drop-on-latency=true ! queue ! rtph264depay ! h264parse ! mppvideodec  "
            f"!  videorate ! video/x-raw,format=NV12,framerate=15/1 ! "
            f"appsink drop=true sync=false"
        )
        return pipeline

    return rtsp


def decode_frames(rtsp, frame_queue, stop_event, max_queue_size=30, call_back_error=None):
    print("bắt đầu luồng giải mã")
    options = {
        'rtsp_transport': 'tcp',
        'stimeout': '5000000',
        'fflags': 'nobuffer',
        'flags': 'low_delay',
        'hwaccel': 'rkmpp',
        'buffer_size': '1024000',  # Tăng buffer size
        'max_delay': '500000',  # Giảm độ trễ tối đa
        'reconnect': '1',  # Tự động kết nối lại nếu mất kết nối
        'reconnect_at_eof': '1',
        'reconnect_streamed': '1',
        'reconnect_delay_max': '5'  # 5 giây tối đa cho việc kết nối lại
    }
    container = None
    try:
        container = av.open(rtsp, options=options)
        video_stream = container.streams.video[0]
        for frame in container.decode(video_stream):
            # print("Đã giải mã frame")
            if stop_event.is_set():
                break

            # Nếu queue đầy, loại bỏ frame cũ nhất
            if frame_queue.qsize() >= max_queue_size:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put(frame)

            # Giảm tần suất giải mã nếu queue gần đầy
            if frame_queue.qsize() > max_queue_size * 0.8:
                time.sleep(0.01)
    except Exception as e:
        print(f"Lỗi giải mã: {e}")
        status_code = 500
        mess = str(e)
        if call_back_error:
            call_back_error(rtsp, status_code, mess)
    finally:
        stop_event.set()
        if container:
            try:
                container.close()
            except Exception as e:
                print(f"Lỗi khi đóng container: {e}")

    print("kết thúc luồng giải mã")


def get_model_plate_platform(platform=None) -> dict[str, str | None]:
    if platform is None:
        platform = get_os_name()
    if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
        return {"model_path": "app/weights/plate_number.rknn", "target": "rk3588", "device_id": None}

    return {"model_path": "app/weights/plate_number.onnx"}
