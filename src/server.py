import asyncio
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.broadcaster import ConnectionManager


app = FastAPI(title="SwarmRL WebSocket Server")

manager = ConnectionManager()


@app.get("/")
async def root():
    return {
        "message": "SwarmRL WebSocket Server is running"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_state():
    drone_states = [
        {
            "drone_id": "drone_1",
            "x": 10.0,
            "y": 20.0,
            "z": 5.0,
            "battery": 95.0,
            "status": "active"
        },
        {
            "drone_id": "drone_2",
            "x": 15.0,
            "y": 25.0,
            "z": 8.0,
            "battery": 90.0,
            "status": "active"
        }
    ]

    while True:
        state = {
            "timestamp": datetime.now().isoformat(),
            "drones": drone_states
        }

        await manager.broadcast(state)

        await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_state())