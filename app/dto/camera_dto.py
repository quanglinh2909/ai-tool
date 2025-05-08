from typing import Any, Optional

from pydantic import BaseModel


class DrawBoundingBoxDTO(BaseModel):
    data: Any
    angle: int
    camera_id: int
class AiDTO(BaseModel):
    is_detect_plate: bool
    is_detect_face: bool
    is_direction_face: bool
    is_direction_plate: bool
    direction_angle_face : int
    direction_angle_plate : int
    direction_deviation_face : int
    direction_deviation_plate : int
    points_face: Optional[list] = []
    points_plate: Optional[list] = []

class CameraDTO(BaseModel):
    name: str
    ip: str
    htt_port: int
    username: str
    password: str
    rtsp: str
    setting: Optional[AiDTO] = None

class GetRtspDTO(BaseModel):
    ip: str
    htt_port: int
    username: str
    password: str

class CameraUpdateDTO(BaseModel):
    name: Optional[str]= None
    ip: Optional[str] = None
    htt_port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    rtsp: Optional[str] = None

