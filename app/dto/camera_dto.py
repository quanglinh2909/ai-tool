from typing import Any, Optional

from pydantic import BaseModel


class DrawBoundingBoxDTO(BaseModel):
    data: Any
    angle: int
    camera_id: int

class CameraDTO(BaseModel):
    name: str
    ip: str
    htt_port: int
    username: str
    password: str
    rtsp: str

class CameraUpdateDTO(BaseModel):
    name: Optional[str]= None
    ip: Optional[str] = None
    htt_port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    rtsp: Optional[str] = None

