"""VERITAS affine geometry verification tests (P0.4 migration).

These tests certify AFFINE-ONLY: synthetic correspondences are generated under
a known affine transform with injected outliers.
"""

import numpy as np

from veritas.geometry import (
    AffineVerificationConfig,
    apply_transformation,
    compute_error_statistics,
    compute_reprojection_errors,
    verify_affine,
    verify_affine_from_correspondences,
)
from veritas.matching import MatchResult

KNOWN_AFFINE = np.array(
    [
        [0.99, -0.04, 12.0],
        [0.04, 0.99, -8.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _project(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return apply_transformation(points, transform)


def test_affine_recovers_known_transform_with_outliers():
    source = np.array(
        [[0, 0], [30, 0], [0, 40], [30, 40], [15, 20], [50, 10], [60, 5], [70, 55]],
        dtype=np.float64,
    )
    reference = _project(source, KNOWN_AFFINE)
    reference[-2:] = [[300.0, -100.0], [-250.0, 400.0]]  # injected outliers

    result = verify_affine(
        source, reference,
        AffineVerificationConfig(reprojection_threshold=1.0, min_inliers=4, random_seed=7),
    )

    assert result.diagnostics.model_name == "affine"
    assert result.is_valid
    assert result.inlier_mask.sum() == 6
    assert np.allclose(result.transformation, KNOWN_AFFINE, atol=1e-6)


def test_affine_rejects_injected_outliers():
    rng = np.random.default_rng(2)
    source = rng.uniform(10, 190, (30, 2))
    reference = _project(source, KNOWN_AFFINE)
    reference[-8:] = rng.uniform(0, 200, (8, 2))

    result = verify_affine(
        source, reference,
        AffineVerificationConfig(reprojection_threshold=0.1, min_inliers=15, random_seed=3),
    )
    assert result.is_valid
    assert result.inlier_mask.sum() == 22


def test_affine_recovers_shear_and_scale():
    transform = np.array(
        [[1.05, 0.12, 33.0], [-0.08, 0.92, -21.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(17)
    source = rng.uniform(20, 300, (40, 2))
    reference = _project(source, transform) + rng.normal(0, 0.15, source.shape)

    result = verify_affine(source, reference, AffineVerificationConfig(reprojection_threshold=0.5, random_seed=5))
    assert result.is_valid
    assert result.inlier_mask.sum() == 40
    # Noise sigma 0.15 px propagates to ~0.15 px in the translation estimate.
    assert np.allclose(result.transformation, transform, atol=0.5)


def test_insufficient_points_returns_invalid():
    source = np.array([[0.0, 0.0], [5.0, 5.0]])
    reference = source + 1.0
    result = verify_affine(source, reference)
    assert not result.is_valid
    assert result.transformation is None


def test_mismatched_point_counts_raise():
    source = np.zeros((5, 2))
    reference = np.zeros((6, 2))
    try:
        verify_affine(source, reference)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_collinear_samples_do_not_certify():
    source = np.array(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0], [50.0, 0.0]],
        dtype=np.float64,
    )
    reference = _project(source, KNOWN_AFFINE)
    result = verify_affine(source, reference, AffineVerificationConfig(random_seed=9))
    assert not result.is_valid  # degenerate collinear samples cannot be certified


def test_error_statistics_matches_manual_computation():
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    stats = compute_error_statistics(errors)
    assert np.isclose(stats.rmse, np.sqrt(np.mean(errors ** 2)))
    assert np.isclose(stats.median, np.median(errors))
    assert np.isclose(stats.percentile_95, np.percentile(errors, 95))


def test_reprojection_errors_are_euclidean():
    source = np.array([[0.0, 0.0], [10.0, 20.0]])
    reference = source + [2.0, -1.0]
    errors = compute_reprojection_errors(
        source, reference, np.eye(3)
    )
    assert np.allclose(errors, np.hypot(2.0, 1.0))


def test_verify_affine_from_correspondences_wraps_match_result():
    source = np.array([[0, 0], [30, 0], [0, 40], [30, 40], [15, 20]], dtype=np.float64)
    reference = _project(source, KNOWN_AFFINE)
    match_result = MatchResult(
        source_points=source.astype(np.float32),
        reference_points=reference.astype(np.float32),
        candidate_count=len(source),
        accepted_count=len(source),
    )
    result = verify_affine_from_correspondences(
        match_result, AffineVerificationConfig(reprojection_threshold=1.0, random_seed=11)
    )
    assert result.is_valid
    assert result.inlier_mask.sum() == 5
    assert result.get_transformation_2x3().shape == (2, 3)


def test_certificate_records_affine_evidence_without_a_confidence_score():
    source = np.array([[0, 0], [30, 0], [0, 40], [30, 40], [15, 20]], dtype=np.float64)
    result = verify_affine(source, _project(source, KNOWN_AFFINE))
    payload = result.to_dict()
    assert result.model == "affine"
    assert result.candidate_count == 5
    assert result.inlier_count == 5
    assert result.p95_error >= 0.0
    assert "confidence" not in payload
