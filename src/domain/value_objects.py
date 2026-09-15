from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobId:
    value: str


@dataclass(frozen=True)
class Confidence:
    value: float


@dataclass(frozen=True)
class ModelVersion:
    coarse: str
    detail: str
