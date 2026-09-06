import numpy as np
from veritas.counter_evidence import summarize_feature_disagreement, summarize_residuals, summarize_spatial_concentration
from veritas.schemas import QuorumEvidence

def test_counter_evidence_reports_actual_residual_and_concentration_measurements():
    residuals = summarize_residuals([0.1, 3.0], 1.0)
    assert residuals["threshold_exceedance_count"] == 1
    assert residuals["max_reprojection_error"] == residuals["maximum"] == 3.0
    assert residuals["mean_reprojection_error"] == residuals["mean"]
    result = summarize_spatial_concentration({"occupied_cells": 1, "total_cells": 16, "coverage_ratio": 1/16}, {"normalized_entropy": 0.0})
    assert result["concentrated"] is True


def test_spatial_concentration_exposes_dominant_cell_fraction():
    coverage = {"occupied_cells": 2, "total_cells": 16, "coverage_ratio": 2 / 16,
                "points_per_cell": np.array([[8, 2], [0, 0]])}
    result = summarize_spatial_concentration(coverage, {"normalized_entropy": 0.18})
    assert result["dominant_cell_count"] == 8
    assert result["dominant_cell_fraction"] == 0.8
    assert result["concentration_score"] == 0.8
    assert result["concentrated"] is True


def test_feature_disagreement_exposes_a_normalized_score():
    quorum = QuorumEvidence(
        detector_results={"sift": "supporting", "orb": "disagreeing", "akaze": "supporting"},
        available_detectors=["sift", "orb", "akaze"],
        supporting_detectors=["sift", "akaze"],
        disagreeing_detectors=["orb"],
        quorum_strength=2 / 3,
    )
    assert summarize_feature_disagreement(quorum)["disagreement_score"] == 1 / 3
