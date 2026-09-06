"""VERITAS verification-metrics tests (P0.5 migration)."""

import numpy as np

from veritas.spatial import calculate_spatial_coverage
from veritas.verification import (
    EvaluationThresholds,
    calculate_error_statistics,
    calculate_inlier_ratio,
    calculate_reprojection_errors,
    calculate_rmse,
    evaluate_registration,
)


def test_inlier_ratio():
    mask = np.array([True, True, False, True])
    assert np.isclose(calculate_inlier_ratio(mask), 0.75)
    assert calculate_inlier_ratio(np.empty(0)) == 0.0


def test_reprojection_errors_match_manual_translation():
    rng = np.random.default_rng(3)
    source = rng.uniform(0, 100, (30, 2))
    transform = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    reference = source + [2.0, -1.0] + rng.normal(0, 0.2, source.shape)
    errors = calculate_reprojection_errors(source, reference, transform)
    assert errors.shape == (30,)
    assert np.allclose(np.mean(errors), np.mean(np.linalg.norm(reference - source - [2.0, -1.0], axis=1)))


def test_rmse_and_error_statistics():
    errors = np.array([1.0, 2.0, 3.0])
    assert np.isclose(calculate_rmse(errors), np.sqrt((1 + 4 + 9) / 3.0))
    stats = calculate_error_statistics(errors)
    assert stats["count"] == 3
    assert np.isclose(stats["median"], 2.0)
    assert np.isclose(stats["rmse"], calculate_rmse(errors))
    assert np.isclose(stats["p95"], np.percentile(errors, 95))
    empty = calculate_error_statistics(np.empty(0))
    assert empty["count"] == 0
    assert np.isnan(empty["rmse"])


def test_spatial_coverage_full_and_degenerate():
    image_shape = (100, 100)
    spread = np.array([[0, 0], [99, 99], [50, 50], [25, 75]], dtype=np.float64)
    full = calculate_spatial_coverage(spread, image_shape)
    assert full["total_cells"] == 16
    assert full["occupied_cells"] <= 16
    assert 0.0 <= full["coverage_percentage"] <= 100.0
    degenerate = calculate_spatial_coverage(np.array([[50.0, 50.0]]), image_shape)
    assert degenerate["occupied_cells"] == 1


def test_spatial_coverage_ignores_out_of_bounds():
    points = np.array([[50.0, 50.0], [-5.0, 50.0], [50.0, 500.0], [np.nan, 50.0]])
    coverage = calculate_spatial_coverage(points, (100, 100))
    assert coverage["occupied_cells"] == 1


def test_evaluate_registration_passes_and_fails():
    rng = np.random.default_rng(5)
    source = rng.uniform(10, 90, (20, 2))
    transform = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    reference = source + [3.0, -2.0] + rng.normal(0, 0.1, source.shape)
    mask = np.ones(20, dtype=bool)

    report = evaluate_registration(source, reference, transform, mask, image_shape=(100, 100))
    assert report["candidate_matches"] == 20
    assert report["verified_inliers"] == 20
    assert report["outliers"] == 0
    assert report["passed"] is True
    assert "spatial_coverage" in report
    assert report["error_histogram"].shape[0] == 180
    assert report["residual_vector_visualization"].shape == (100, 100, 3)
    assert report["metric_scope"].startswith("image-based")

    # A grossly wrong transform must fail the report.
    bad_transform = np.array([[1.0, 0.0, 500.0], [0.0, 1.0, -500.0], [0.0, 0.0, 1.0]])
    bad_report = evaluate_registration(source, reference, bad_transform, mask)
    assert bad_report["passed"] is False


def test_evaluate_registration_respects_custom_thresholds():
    source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    reference = source + [0.5, 0.0]
    transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mask = np.ones(4, dtype=bool)
    strict = evaluate_registration(source, reference, transform, mask, thresholds={"max_rmse_px": 0.1, "min_inlier_ratio": 0.5})
    assert strict["passed"] is False
    lenient = evaluate_registration(source, reference, transform, mask, thresholds=EvaluationThresholds(max_rmse_px=1.0))
    assert lenient["passed"] is True