import numpy as np
from veritas.counter_evidence import summarize_residuals, summarize_spatial_concentration

def test_counter_evidence_reports_actual_residual_and_concentration_measurements():
    assert summarize_residuals([0.1, 3.0], 1.0)["threshold_exceedance_count"] == 1
    result = summarize_spatial_concentration({"occupied_cells": 1, "total_cells": 16, "coverage_ratio": 1/16}, {"normalized_entropy": 0.0})
    assert result["concentrated"] is True


def test_spatial_concentration_exposes_dominant_cell_fraction():
    coverage = {"occupied_cells": 2, "total_cells": 16, "coverage_ratio": 2 / 16,
                "points_per_cell": np.array([[8, 2], [0, 0]])}
    result = summarize_spatial_concentration(coverage, {"normalized_entropy": 0.18})
    assert result["dominant_cell_count"] == 8
    assert result["dominant_cell_fraction"] == 0.8
    assert result["concentrated"] is True
