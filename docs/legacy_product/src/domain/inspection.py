"""Product inspection policy shared by CLI, API and video processing."""
from __future__ import annotations

import math
from src.domain.entities import ClassificationResult

DEFECT_LABELS = ("DENT", "SCRATCH", "CRACK", "DIRT", "MISSING_PART", "MISSING_LABEL", "MISALIGNED_LABEL")
DEFAULT_CONFIG = "configs/product_inspection.yml"
DEFAULT_WEIGHTS = "ckpts/product_inspection.pth"


def inspection_result(scores: dict[str, float], review_threshold: float = 0.3,
                      fail_threshold: float = 0.7, criteria: int = 0) -> ClassificationResult:
    if not 0 < review_threshold < fail_threshold < 1:
        raise ValueError("Require 0 < review_threshold < fail_threshold < 1")
    if not scores or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
        raise ValueError("Defect scores must be non-empty finite probabilities in [0, 1]")
    label = max(scores, key=scores.get)
    risk = scores[label]
    decision = "FAIL" if risk >= fail_threshold else "REVIEW" if risk >= review_threshold else "PASS"
    return ClassificationResult(
        coarse_label=decision,
        coarse_score=risk,  # maximum defect probability, NOT decision confidence
        detail_label=label if decision != "PASS" else None,
        detail_score=risk if decision != "PASS" else None,
        meta={"decision": decision, "defect_score": risk, "scores": dict(scores),
              "defects": [k for k, p in scores.items() if p >= fail_threshold],
              "suspected_defects": [k for k, p in scores.items() if review_threshold <= p < fail_threshold],
              "review_threshold": review_threshold, "fail_threshold": fail_threshold,
              "criteria": criteria},
    )


def inspection_payload(result: ClassificationResult) -> dict:
    return {**result.meta, "decision": result.coarse_label, "defect_score": result.coarse_score}


def aggregate_frames(results: list[ClassificationResult], paths: list[str]) -> ClassificationResult:
    if not results or len(results) != len(paths):
        raise ValueError("Video inspection requires matching non-empty frames and results")
    scores = {name: max(r.meta["scores"][name] for r in results) for name in results[0].meta["scores"]}
    first = results[0].meta
    result = inspection_result(scores, first["review_threshold"], first["fail_threshold"], first.get("criteria", 0))
    worst = max(range(len(results)), key=lambda i: results[i].coarse_score)
    result.meta.update(frame_count=len(results), worst_frame=paths[worst],
                       aggregation="max_per_defect_over_sampled_frames")
    return result
