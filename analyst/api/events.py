"""Event bus: how the worker's progress reaches WebSocket clients.

Two backends behind one interface (the same pattern as LLM providers and sandboxes —
by now this is the house style):

- RedisEventBus     — production: worker process publishes, API process subscribes.
  Survives the worker and API being different processes/containers (phase g).
- InProcessEventBus — dev/tests: plain asyncio queues. Lets the whole system run as
  ONE process with zero infrastructure (ANALYST_DEV_INPROCESS=1), which is also how
  the API test suite runs in CI.

Events are also APPENDED to the run store as they happen, so a client that connects
late (or reconnects) replays history first, then goes live — streaming is UX,
the store is truth (ARCHITECTURE.md §5).
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from typing import AsyncIterator, Protocol

DONE_SENTINEL = {"type": "stream_end"}


class EventBus(Protocol):
    async def publish(self, run_id: str, event: dict) -> None: ...
    def subscribe(self, run_id: str) -> AsyncIterator[dict]: ...


class InProcessEventBus:
    def __init__(self):
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, run_id: str, event: dict) -> None:
        for q in self._queues.get(run_id, []):
            q.put_nowait(event)

    async def subscribe(self, run_id: str) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[run_id].append(q)
        try:
            while True:
                event = await q.get()
                yield event
                if event.get("type") == "stream_end":
                    return
        finally:
            self._queues[run_id].remove(q)


class RedisEventBus:
    def __init__(self, url: str | None = None):
        import redis.asyncio as aioredis

        self.redis = aioredis.from_url(
            url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        )

    @staticmethod
    def _channel(run_id: str) -> str:
        return f"analyst:run:{run_id}"

    async def publish(self, run_id: str, event: dict) -> None:
        await self.redis.publish(self._channel(run_id), json.dumps(event, default=str))

    async def subscribe(self, run_id: str) -> AsyncIterator[dict]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._channel(run_id))
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                event = json.loads(msg["data"])
                yield event
                if event.get("type") == "stream_end":
                    return
        finally:
            await pubsub.unsubscribe(self._channel(run_id))
            await pubsub.aclose()


def make_event_bus() -> EventBus:
    if os.environ.get("ANALYST_DEV_INPROCESS"):
        return InProcessEventBus()
    return RedisEventBus()
