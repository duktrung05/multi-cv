from __future__ import annotations

import asyncio
from collections import defaultdict


class InMemoryStageQueue:
    def __init__(self, max_size: int = 4096) -> None:
        self._max_size = max_size
        self._queues: dict[str, asyncio.Queue[str]] = defaultdict(self._new_queue)

    def _new_queue(self) -> asyncio.Queue[str]:
        return asyncio.Queue(maxsize=self._max_size)

    async def publish(self, stage: str, job_id: str) -> None:
        await self._queues[stage].put(job_id)

    async def consume(self, stage: str) -> str:
        return await self._queues[stage].get()
