from typing import Any

from piccolo.columns import ForeignKey, Boolean, Integer, Array, Varchar,JSON

from app.constants.type_ai_enum import TypeAIEnum
from app.models.base_model import BaseModel
from app.models.camera_model import Camera

class ProcessAI(BaseModel):
    camera = ForeignKey(references=Camera, db_column_name="camera_id")
    direction = Integer(default=-1)
    points = JSON(default=list)
    type_ai = Varchar(length=20, choices=TypeAIEnum)

