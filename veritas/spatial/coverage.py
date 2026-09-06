"""VERITAS spatial — coverage statistics over verified correspondences.

P0.5: migrated from the reference metrics module. ``calculate_spatial_coverage``
returns descriptive occupancy statistics (how many cells of a fixed 4x4 grid the
verified points occupy); it is not a claim about scene geometry.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def calculate_spatial_coverage(points: Any, image_shape: Any) -> Dict[str, Any]:
    """Independent 4x4 occupancy coverage for evaluation reports."""
    points = np.asarray(points, dtype=np.float64)
    h, w = int(image_shape[0]), int(image_shape[1])
    counts = np.zeros((4, 4), dtype=int)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0) & (points[:, 0] < w)
        & (points[:, 1] >= 0) & (points[:, 1] < h)
    )
    for x, y in points[valid]:
        counts[min(int(y * 4 / h), 3), min(int(x * 4 / w), 3)] += 1
    return {
        "occupied_cells": int(np.count_nonzero(counts)),
        "total_cells": 16,
        "coverage_percentage": 100.0 * np.count_nonzero(counts) / 16,
        "points_per_cell": counts,
    }