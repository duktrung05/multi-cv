"""Image predictions and sampled-frame aggregation for AI face manipulation."""
from src.domain.entities import ClassificationResult

DEFAULT_CONFIG = "configs/dgm4_binary.yml"
DEFAULT_WEIGHTS = "ckpts/dgm4_binary.pth"


def classification_result(prediction, criteria=0):
    return ClassificationResult(coarse_label=prediction["label"], coarse_score=prediction["score"],
                                meta={**prediction, "criteria": criteria})


def classification_payload(result):
    return {**result.meta, "label": result.coarse_label, "score": result.coarse_score}


def aggregate_frames(results, paths):
    if not results or len(results) != len(paths):
        raise ValueError("Require matching non-empty frames and results")
    threshold = results[0].meta["ai_edited_threshold"]
    if any(r.meta["ai_edited_threshold"] != threshold for r in results):
        raise ValueError("Frame thresholds must match")
    worst = max(range(len(results)), key=lambda i: results[i].meta["scores"]["AI_EDITED"])
    selected = results[worst]
    return ClassificationResult(coarse_label=selected.coarse_label, coarse_score=selected.coarse_score,
        meta={**selected.meta, "frame_count": len(results), "worst_frame": paths[worst],
              "aggregation": "maximum_ai_edited_score_over_sampled_frames",
              "scope": "Sampled frames only; video-level accuracy has not been validated"})
