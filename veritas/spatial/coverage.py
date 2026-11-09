"""VERITAS spatial — coverage statistics over verified correspondences.

P0.5: migrated from the reference metrics module. ``calculate_spatial_coverage``
returns descriptive occupancy statistics (how many cells of a fixed 4x4 grid the
verified points occupy); it is not a claim about scene geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class CoverageConfig:
    """Grid configuration; the Phase 1 default is 4 by 4 cells."""

    grid_shape: Tuple[int, int] = (4, 4)

    def __post_init__(self) -> None:
        rows, columns = self.grid_shape
        if rows <= 0 or columns <= 0:
            raise ValueError("grid_shape dimensions must be positive")


def calculate_spatial_coverage(
    points: Any,
    image_shape: Any,
    config: CoverageConfig | None = None,
) -> Dict[str, Any]:
    """Return raw configurable-grid occupancy evidence for verified points."""
    points = np.asarray(points, dtype=np.float64)
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("image_shape dimensions must be positive")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    settings = config or CoverageConfig()
    rows, columns = settings.grid_shape
    counts = np.zeros((rows, columns), dtype=int)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0) & (points[:, 0] < w)
        & (points[:, 1] >= 0) & (points[:, 1] < h)
    )
    for x, y in points[valid]:
        row = min(int(y * rows / h), rows - 1)
        column = min(int(x * columns / w), columns - 1)
        counts[row, column] += 1

    occupied = int(np.count_nonzero(counts))
    total = rows * columns
    return {
        "grid_shape": settings.grid_shape,
        "occupied_cells": occupied,
        "total_cells": total,
        "coverage_ratio": occupied / total,
        "coverage_percentage": 100.0 * occupied / total,
        "points_per_cell": counts,
        "configuration": asdict(settings),
    }
