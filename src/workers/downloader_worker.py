from __future__ import annotations

import asyncio

from src.bootstrap import build_container
from src.shared.config import Settings


async def main() -> None:
    container = build_container(Settings.from_env())
    await container.workers.run_download_worker()


if __name__ == "__main__":
    asyncio.run(main())
