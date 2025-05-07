from typing import Any

from pydantic import BaseModel


class DrawBoundingBoxDTO(BaseModel):
    data: Any
    angle: int
    camera_id: int


