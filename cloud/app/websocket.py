"""WebSocket shared memory — real-time sync between agents."""
import json
import asyncio
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
from app.config import REDIS_URL


class ConnectionManager:
    """Manages WebSocket connections and Redis pub/sub for cross-process broadcast."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}  # pool_id -> set of websockets
        self._redis = None
        self._subscriber_task = None

    async def get_redis(self):
        if self._redis is None:
            self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        return self._redis

    async def start_subscriber(self):
        """Background task: listen to Redis pub/sub and broadcast to local WebSocket clients."""
        r = await self.get_redis()
        pubsub = r.pubsub()
        await pubsub.psubscribe("memchip:memory:*")
        async for message in pubsub.listen():
            if message["type"] == "pmessage":
                channel = message["channel"]  # memchip:memory:<pool_id>
                pool_id = channel.split(":", 2)[2] if channel.count(":") >= 2 else "default"
                data = message["data"]
                await self._broadcast_local(pool_id, data)

    async def _broadcast_local(self, pool_id: str, data: str):
        """Send to all local WebSocket connections subscribed to this pool."""
        if pool_id in self.connections:
            dead = set()
            for ws in self.connections[pool_id]:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.add(ws)
            self.connections[pool_id] -= dead

    async def connect(self, websocket: WebSocket, pool_ids: list[str]):
        await websocket.accept()
        for pid in pool_ids:
            if pid not in self.connections:
                self.connections[pid] = set()
            self.connections[pid].add(websocket)

    def disconnect(self, websocket: WebSocket):
        for pool_id in list(self.connections.keys()):
            self.connections[pool_id].discard(websocket)
            if not self.connections[pool_id]:
                del self.connections[pool_id]

    async def publish_event(self, pool_id: str, event_type: str, data: dict):
        """Publish memory event to Redis (broadcasts to all API instances)."""
        r = await self.get_redis()
        payload = json.dumps({"event": event_type, "pool_id": pool_id, "data": data})
        await r.publish(f"memchip:memory:{pool_id}", payload)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """
    WS /v1/ws?pools=personal,shared:team&token=mc_xxx

    Client sends:
      {"action": "subscribe", "pools": ["shared:team", "agent:luna"]}
      {"action": "unsubscribe", "pools": ["shared:team"]}

    Server sends:
      {"event": "memory.added", "pool_id": "shared:team", "data": {...}}
      {"event": "memory.updated", ...}
      {"event": "memory.deleted", ...}
    """
    # Extract pools from query params
    pools = websocket.query_params.get("pools", "default").split(",")
    await manager.connect(websocket, pools)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "subscribe":
                    new_pools = msg.get("pools", [])
                    for pid in new_pools:
                        if pid not in manager.connections:
                            manager.connections[pid] = set()
                        manager.connections[pid].add(websocket)
                    await websocket.send_text(json.dumps({"event": "subscribed", "pools": new_pools}))
                elif action == "unsubscribe":
                    for pid in msg.get("pools", []):
                        if pid in manager.connections:
                            manager.connections[pid].discard(websocket)
                    await websocket.send_text(json.dumps({"event": "unsubscribed", "pools": msg.get("pools", [])}))
                elif action == "ping":
                    await websocket.send_text(json.dumps({"event": "pong"}))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"event": "error", "message": "Invalid JSON"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
