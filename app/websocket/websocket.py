from fastapi import APIRouter, Query
from starlette.websockets import WebSocket, WebSocketDisconnect
from typing import List

from app.websocket.ConnectionManager import connection_manager

router = APIRouter()


@router.websocket("/stream")
async def websocket_super_admin(websocket: WebSocket,rtsp):

    await connection_manager.connect(websocket, rtsp)
    try:
        while True:
            data = await websocket.receive_json()
            # Xử lý các action khác nếu cần
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)



