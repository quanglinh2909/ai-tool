from piccolo.columns import Varchar,Integer
from app.models.base_model import BaseModel


class Camera(BaseModel):
    name = Varchar(length=100)
    ip = Varchar(length=100)
    htt_port = Integer()
    username = Varchar(length=100)
    password = Varchar(length=100)
    rtsp = Varchar(length=100)

