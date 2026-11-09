"""VERITAS verification — image-based registration evaluation metrics.

P0.5 — migrated from the reference repository's metrics module. These are
image-based registration metrics (reprojection error, inlier ratio, RMSE,
spatial coverage); they are explicitly not geographic ground-truth accuracy.

The private renderers ``_histogram`` and ``_create_residual_vector_visualization``
are migrated here because the evaluation report embeds them; a first-class audit
renderer module is P1 work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..spatial.coverage import calculate_spatial_coverage


@dataclass
class EvaluationThresholds:
    """Pass/fail thresholds for the image-based registration report."""

    max_rmse_px: float = 3.0
    min_inlier_ratio: float = 0.5


def calculate_inlier_ratio(inlier_mask: Any) -> float:
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    return float(np.mean(mask)) if len(mask) else 0.0


def _as_matrix(transformation: Any) -> np.ndarray:
    matrix = np.asarray(transformation, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, (0.0, 0.0, 1.0)))
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("transformation must be a finite 2x3 or 3x3 matrix")
    return matrix


def calculate_reprojection_errors(
    source_points: Any, reference_points: Any, transformation: Any
) -> np.ndarray:
    source, reference = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or reference.shape != source.shape:
        raise ValueError("point arrays must have matching shape (N,2)")
    matrix = _as_matrix(transformation)
    projected = (matrix @ np.c_[source, np.ones(len(source))].T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("transformation projects a point to infinity")
    return np.linalg.norm(projected[:, :2] / projected[:, 2:3] - reference, axis=1)


def calculate_rmse(errors: Any) -> float:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


def calculate_error_statistics(errors: Any) -> Dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "rmse": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": calculate_rmse(values),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _histogram(errors: Any, width: int = 480, height: int = 180) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, np.uint8)
    if len(errors):
        counts, _ = np.histogram(errors, bins=20)
        scale = (height - 20) / max(int(counts.max()), 1)
        for i, count in enumerate(counts):
            x0, x1 = i * width // 20, (i + 1) * width // 20 - 1
            cv2.rectangle(canvas, (x0, height - 1), (x1, height - round(count * scale)), (200, 80, 20), -1)
    return canvas


def _create_residual_vector_visualization(
    source_points: Any,
    reference_points: Any,
    transformation: Any,
    image_shape: Any,
    inlier_mask: Any = None,
) -> np.ndarray:
    """Render predicted-to-observed residual vectors in the reference frame."""
    source, reference = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    matrix = _as_matrix(transformation)
    h, w = int(image_shape[0]), int(image_shape[1])
    predicted = (matrix @ np.c_[source, np.ones(len(source))].T).T
    predicted = predicted[:, :2] / predicted[:, 2:3]
    mask = np.ones(len(source), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool)
    if len(mask) != len(source):
        raise ValueError("inlier_mask length must equal point count")
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for start, end in zip(predicted[mask], reference[mask]):
        if np.isfinite(np.r_[start, end]).all():
            cv2.arrowedLine(canvas, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), (0, 200, 255), 1, tipLength=0.25)
    return canvas


def evaluate_registration(
    source_points: Any,
    reference_points: Any,
    transformation: Any,
    inlier_mask: Any,
    image_shape: Any = None,
    thresholds: Any = None,
) -> Dict[str, Any]:
    """Return image-based metrics; these are explicitly not geographic accuracy."""
    source, reference = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(mask) != len(source):
        raise ValueError("inlier_mask length must equal candidate match count")
    errors = calculate_reprojection_errors(source, reference, transformation)
    inlier_errors = errors[mask]
    stats = calculate_error_statistics(inlier_errors)
    limits = EvaluationThresholds(**thresholds) if isinstance(thresholds, dict) else (thresholds or EvaluationThresholds())
    report: Dict[str, Any] = {
        "metric_scope": "image-based registration metrics; not absolute geographic accuracy",
        "candidate_matches": int(len(source)),
        "verified_inliers": int(mask.sum()),
        "outliers": int((~mask).sum()),
        "inlier_ratio": calculate_inlier_ratio(mask),
        "evaluated_correspondences": "the supplied candidate correspondence arrays and inlier mask",
        "candidate_error_statistics": calculate_error_statistics(errors),
        "inlier_error_statistics": stats,
        "errors": errors,
        "inlier_errors": inlier_errors,
        "error_histogram": _histogram(inlier_errors),
        "thresholds": asdict(limits),
        "passed": bool(stats["rmse"] <= limits.max_rmse_px and calculate_inlier_ratio(mask) >= limits.min_inlier_ratio),
    }
    if image_shape is not None:
        report["spatial_coverage"] = calculate_spatial_coverage(source[mask], image_shape)
        report["residual_vector_visualization"] = _create_residual_vector_visualization(
            source, reference, transformation, image_shape, mask
        )
    return report