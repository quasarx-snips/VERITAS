"""Spatial concentration facts derived from coverage and entropy.

Reports the dominant grid cell so that clustered correspondence support is not
concealed by aggregate occupancy numbers.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

#: Fraction of points in the single most-populated cell above which support is
#: considered spatially concentrated.
DOMINANCE_THRESHOLD: float = 0.8


def summarize_spatial_concentration(
    coverage: Mapping[str, Any],
    entropy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Summarize spatial concentration from coverage and entropy measurements.

    Args:
        coverage: Occupancy statistics from ``calculate_spatial_coverage``.
        entropy: Entropy statistics from ``calculate_spatial_entropy``.

    Returns:
        A dictionary describing the dominant cell, its fraction of points, and a
        ``concentrated`` flag.
    """
    counts = np.asarray(coverage.get("points_per_cell", ()), dtype=float)
    total = float(counts.sum())
    dominant_count = int(counts.max()) if counts.size else 0
    dominant_fraction: Optional[float] = dominant_count / total if total else None
    concentrated = bool(
        coverage["occupied_cells"] <= 1
        or (dominant_fraction is not None and dominant_fraction >= DOMINANCE_THRESHOLD)
    )
    return {
        "occupied_cells": coverage["occupied_cells"],
        "total_cells": coverage["total_cells"],
        "coverage_ratio": coverage["coverage_ratio"],
        "normalized_entropy": entropy["normalized_entropy"],
        "dominant_cell_count": dominant_count,
        "dominant_cell_fraction": dominant_fraction,
        "concentration_score": dominant_fraction if dominant_fraction is not None else 0.0,
        "concentrated": concentrated,
    }
