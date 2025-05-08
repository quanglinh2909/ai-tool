from app.constants.platform_enum import PlatformEnum
from app.ultils.check_platform import get_os_name


def get_rtsp_platform(rtsp: str,platform=None) -> str:
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

def get_model_plate_platform(platform=None) -> str:
    if platform is None:
        platform = get_os_name()
    if platform == PlatformEnum.ORANGE_PI_MAX or platform == PlatformEnum.ORANGE_PI:
        return 'app/weights/vehicle_plate_2_rknn_model'
    return 'app/weights/vehicle_plate_2.onnx'
