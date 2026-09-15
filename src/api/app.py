from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from src.api.routers.health import router as health_router
from src.api.routers.jobs import router as jobs_router
from src.api.routers.test_classify import router as test_classify_router
from src.bootstrap import build_container
from src.shared.config import Settings


def create_app(container_builder: Callable = build_container) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = Settings.from_env()
        container = container_builder(settings)
        app.state.container = container
        await container.workers.start()
        try:
            yield
        finally:
            await container.workers.stop()

    app = FastAPI(title="reas-iasvas-add-multilabel-classification API", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(test_classify_router)
    return app


app = create_app()
