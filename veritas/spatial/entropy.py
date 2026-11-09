"""Normalized Shannon entropy over a correspondence grid."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .coverage import CoverageConfig, calculate_spatial_coverage


def calculate_spatial_entropy(
    points: Any,
    image_shape: Any,
    config: CoverageConfig | None = None,
) -> Dict[str, Any]:
    """Compute grid Shannon entropy separately from coverage.

    Args:
        points: Verified correspondence points as an (N, 2) array.
        image_shape: ``(height, width)`` of the image.
        config: Optional grid configuration (defaults to 4x4 cells).

    Returns:
        A dictionary with ``shannon_entropy``, ``normalized_entropy``,
        ``grid_shape``, ``occupied_cells`` and ``coverage_ratio``.
    """
    coverage = calculate_spatial_coverage(points, image_shape, config)
    counts = coverage["points_per_cell"].astype(float).ravel()
    total = counts.sum()
    if not total:
        entropy = 0.0
        normalized = 0.0
    else:
        probabilities = counts[counts > 0] / total
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized = entropy / float(np.log(len(counts))) if len(counts) > 1 else 0.0
    return {
        "shannon_entropy": entropy,
        "normalized_entropy": normalized,
        "grid_shape": coverage["grid_shape"],
        "occupied_cells": coverage["occupied_cells"],
        "coverage_ratio": coverage["coverage_ratio"],
    }
