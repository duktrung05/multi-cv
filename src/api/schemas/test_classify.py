from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


def _first(v: Any) -> Any:
    if isinstance(v, list) and len(v) > 0:
        return v[0]
    return v


class TestClassifyVideoBody(BaseModel):
    """Tương thích api_vas: url / video_id / criteria có thể là list một phần tử."""

    url: Any
    video_id: Any = "test_video"
    criteria: Any = 0
    frame_step: int | None = Field(default=None, description="Override VIDEO_FRAME_STEP")

    @model_validator(mode="after")
    def normalize(self) -> TestClassifyVideoBody:
        object.__setattr__(self, "url", str(_first(self.url)))
        object.__setattr__(self, "video_id", str(_first(self.video_id)))
        object.__setattr__(self, "criteria", int(_first(self.criteria) if self.criteria is not None else 0))
        return self
