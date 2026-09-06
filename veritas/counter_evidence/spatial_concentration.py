"""Spatial concentration facts derived from coverage and entropy."""

import numpy as np


def summarize_spatial_concentration(coverage, entropy):
    """Report the dominant cell rather than concealing clustered support."""
    counts = np.asarray(coverage.get("points_per_cell", ()), dtype=float)
    total = float(counts.sum())
    dominant_count = int(counts.max()) if counts.size else 0
    dominant_fraction = dominant_count / total if total else None
    concentrated = bool(
        coverage["occupied_cells"] <= 1
        or (dominant_fraction is not None and dominant_fraction >= 0.8)
    )
    return {"occupied_cells": coverage["occupied_cells"], "total_cells": coverage["total_cells"],
            "coverage_ratio": coverage["coverage_ratio"], "normalized_entropy": entropy["normalized_entropy"],
            "dominant_cell_count": dominant_count, "dominant_cell_fraction": dominant_fraction,
            "concentrated": concentrated}
