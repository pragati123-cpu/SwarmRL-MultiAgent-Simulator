import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from terrain_drones import DroneTerrainSimulation

app = FastAPI(title="SwarmRL WebSocket Server")


@app.get("/")
def health_check():
    return {"status": "running"}


@app.websocket("/ws/drones")
async def drone_websocket(websocket: WebSocket):
    await websocket.accept()

    simulation = DroneTerrainSimulation()

    try:
        while True:
            state = simulation.get_simulation_state()

            await websocket.send_text(json.dumps(state))

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")