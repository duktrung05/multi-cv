from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_container
from src.bootstrap import AppContainer

router = APIRouter(tags=["health"])


@router.get("/health")
def health(container: AppContainer = Depends(get_container)) -> dict:
    return {
        "status": "ok",
        "metrics": container.metrics.snapshot(),
    }
