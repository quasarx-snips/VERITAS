"""Unit tests for the VERITAS partial correspondence analysis."""

import pytest
import numpy as np
from unittest.mock import Mock

from veritas.geometry.certificate import AffineCertificate, ErrorStatistics, VerificationDiagnostics
from veritas.matching.descriptor_matching import MatchResult
from veritas.schemas import VerificationEvidence, GeometryEvidence, SpatialEvidence, QuorumEvidence, CounterEvidence
from veritas.verification.partial import (
    analyze_partial_correspondence,
    PartialCorrespondenceConfig,
    PartialCorrespondenceStatus,
    RegionEvidence,
)


@pytest.fixture
def mock_match_result():
    """Fixture for a mock MatchResult."""
    source_points = np.array([
        [10, 10], [20, 20], [30, 30], [40, 40],
        [100, 100], [110, 110], [120, 120], [130, 130],
        [200, 200], [210, 210], [220, 220], [230, 230],
        [10, 200], [20, 210], [30, 220], [40, 230],
    ], dtype=np.float32)
    reference_points = source_points + 1  # Simple translation
    return MatchResult(
        source_points=source_points,
        reference_points=reference_points,
        candidate_count=len(source_points),
        accepted_count=len(source_points),
        filter_diagnostics={},
    )


@pytest.fixture
def affine_certificate_factory():
    """Factory for creating AffineCertificate instances with custom inlier masks."""
    def _factory(source_points, reference_points, inlier_mask):
        inlier_count = np.sum(inlier_mask)
        total_matches = len(source_points)
        inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0

        # Mock a simple transformation that results in small residuals
        transformation = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

        # Create mock ErrorStatistics
        error_stats = ErrorStatistics(
            mean=0.5, median=0.5, std=0.1, rmse=0.5, min_error=0.1, max_error=0.9, percentile_95=0.8
        )

        # Create VerificationDiagnostics
        diagnostics = VerificationDiagnostics(
            model_name="affine",
            total_matches=total_matches,
            inlier_count=inlier_count,
            outlier_count=total_matches - inlier_count,
            inlier_ratio=inlier_ratio,
            iterations_run=1,
            converged=True,
            is_valid=True,
            error_stats=error_stats,
        )

        certificate = AffineCertificate(
            is_valid=True,
            transformation=transformation,
            inlier_mask=inlier_mask,
            diagnostics=diagnostics,
            source_inliers=source_points[inlier_mask],
            reference_inliers=reference_points[inlier_mask],
        )
        # Manually set the compute_reprojection_errors for the mock
        certificate.compute_reprojection_errors = Mock(return_value=np.array([0.5] * inlier_count))
        return certificate
    return _factory


@pytest.fixture
def mock_affine_certificate(mock_match_result, affine_certificate_factory):
    """Fixture for a mock AffineCertificate."""
    inlier_mask = np.array([True] * len(mock_match_result.source_points))
    return affine_certificate_factory(mock_match_result.source_points, mock_match_result.reference_points, inlier_mask)


@pytest.fixture
def mock_verification_evidence():
    """Fixture for a mock VerificationEvidence object."""
    mock_geometry = Mock(spec=GeometryEvidence)
    mock_geometry.certified = True
    mock_geometry.inlier_ratio = 0.8
    mock_geometry.inlier_count = 100
    mock_geometry.to_dict.return_value = {}

    mock_spatial = Mock(spec=SpatialEvidence)
    mock_spatial.coverage_ratio = 0.9
    mock_spatial.normalized_entropy = 0.8
    mock_spatial.to_dict.return_value = {}

    mock_quorum = Mock(spec=QuorumEvidence)
    mock_quorum.quorum_strength = 0.95
    mock_quorum.to_dict.return_value = {}

    mock_counter_evidence = Mock(spec=CounterEvidence)
    mock_counter_evidence.residuals = {"max_reprojection_error": 1.5, "mean_reprojection_error": 0.5}
    mock_counter_evidence.feature_disagreement = {"disagreement_score": 0.1}
    mock_counter_evidence.spatial_concentration = {"concentration_score": 0.1}
    mock_counter_evidence.to_dict.return_value = {}

    evidence = Mock(spec=VerificationEvidence)
    evidence.geometry = mock_geometry
    evidence.spatial = mock_spatial
    evidence.quorum = mock_quorum
    evidence.counter_evidence = mock_counter_evidence
    evidence.to_dict.return_value = {}
    return evidence


@pytest.fixture
def default_partial_config():
    """Fixture for default PartialCorrespondenceConfig."""
    return PartialCorrespondenceConfig(
        grid_shape=(4, 4),
        min_inliers_per_region=1,
        min_inlier_ratio_per_region=0.5,
        max_mean_residual_per_region=1.0,
        max_detector_disagreement_per_region=0.2,
    )


def test_fully_supported_grid(mock_match_result, mock_affine_certificate, mock_verification_evidence):
    """Test case for a fully supported grid."""
    # Create a custom config for this test to ensure full coverage
    custom_config = PartialCorrespondenceConfig(
        grid_shape=(1, 1), # Use a 1x1 grid to ensure all points fall into one region
        min_inliers_per_region=1,
        min_inlier_ratio_per_region=0.5,
        max_mean_residual_per_region=1.0,
        max_detector_disagreement_per_region=0.2,
    )
    image_shape = (250, 250)  # Example image shape
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, custom_config
    )

    assert result.status == PartialCorrespondenceStatus.FULL
    assert len(result.supported_regions) == custom_config.grid_shape[0] * custom_config.grid_shape[1]
    assert not result.unsupported_regions
    assert not result.insufficient_data_regions
    assert result.coverage_fraction == 1.0
    assert len(result.region_evidence) == custom_config.grid_shape[0] * custom_config.grid_shape[1]
    for re in result.region_evidence:
        assert re.status == "SUPPORTED"


def test_partially_supported_grid(mock_match_result, affine_certificate_factory, mock_verification_evidence, default_partial_config):
    """Test case for a partially supported grid."""
    # Modify some inliers to be outliers to create partial support
    modified_inlier_mask = np.array([True] * len(mock_match_result.source_points))
    modified_inlier_mask[0:4] = False  # Make first 4 points outliers
    mock_affine_certificate = affine_certificate_factory(
        mock_match_result.source_points,
        mock_match_result.reference_points,
        modified_inlier_mask
    )

    image_shape = (250, 250)
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, default_partial_config
    )

    assert result.status == PartialCorrespondenceStatus.PARTIAL
    assert len(result.supported_regions) > 0
    assert len(result.unsupported_regions) > 0
    assert result.coverage_fraction < 1.0
    assert result.coverage_fraction > 0.0


def test_insufficient_grid(mock_match_result, affine_certificate_factory, mock_verification_evidence, default_partial_config):
    """Test case for an insufficient grid (no supported regions)."""
    # Make all inliers outliers
    mock_affine_certificate = affine_certificate_factory(
        mock_match_result.source_points,
        mock_match_result.reference_points,
        np.array([False] * len(mock_match_result.source_points))
    )

    image_shape = (250, 250)
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, default_partial_config
    )

    assert result.status == PartialCorrespondenceStatus.INSUFFICIENT
    assert not result.supported_regions
    assert len(result.unsupported_regions) > 0 or len(result.insufficient_data_regions) > 0
    assert result.coverage_fraction == 0.0


def test_unsupported_regions_are_returned(mock_match_result, affine_certificate_factory, mock_verification_evidence, default_partial_config):
    """Test that unsupported regions are correctly identified and returned."""
    # Make some regions unsupported by setting their inliers to 0
    modified_inlier_mask = np.array([True] * len(mock_match_result.source_points))
    # Assuming points 0-3 are in one region, make them outliers
    modified_inlier_mask[0:4] = False
    mock_affine_certificate = affine_certificate_factory(
        mock_match_result.source_points,
        mock_match_result.reference_points,
        modified_inlier_mask
    )

    image_shape = (250, 250)
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, default_partial_config
    )

    assert len(result.unsupported_regions) > 0
    # Check if the specific region (0,0) is unsupported due to insufficient inliers
    assert (0, 0) in result.unsupported_regions
    assert result.local_failure_reasons[(0, 0)] == ["INSUFFICIENT_INLIERS", "LOW_INLIER_RATIO"]


def test_region_evidence_is_serializable(mock_match_result, affine_certificate_factory, mock_verification_evidence, default_partial_config):
    """Test that region evidence is serializable to a dictionary."""
    image_shape = (250, 250)
    mock_affine_certificate = affine_certificate_factory(
        mock_match_result.source_points,
        mock_match_result.reference_points,
        np.array([True] * len(mock_match_result.source_points))
    )
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, default_partial_config
    )

    for re in result.region_evidence:
        assert isinstance(re.to_dict(), dict)
        assert "status" in re.to_dict()
        assert "inlier_count" in re.to_dict()
        assert "row" in re.to_dict()
        assert "col" in re.to_dict()


def test_no_fabricated_regions(mock_match_result, affine_certificate_factory, mock_verification_evidence, default_partial_config):
    """Test that no fabricated regions are returned."""
    image_shape = (250, 250)
    mock_affine_certificate = affine_certificate_factory(
        mock_match_result.source_points,
        mock_match_result.reference_points,
        np.array([True] * len(mock_match_result.source_points))
    )
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, default_partial_config
    )

    total_regions_in_grid = default_partial_config.grid_shape[0] * default_partial_config.grid_shape[1]
    assert len(result.region_evidence) == total_regions_in_grid
    assert len(result.supported_regions) + len(result.unsupported_regions) + len(result.insufficient_data_regions) == total_regions_in_grid


def test_partial_correspondence_config_values(mock_match_result, affine_certificate_factory, mock_verification_evidence):
    """Test that partial correspondence respects config values."""
    custom_config = PartialCorrespondenceConfig(
        grid_shape=(2, 2),
        min_inliers_per_region=5,
        min_inlier_ratio_per_region=0.8,
        max_mean_residual_per_region=0.1,
        max_detector_disagreement_per_region=0.05,
    )
    image_shape = (250, 250)
    mock_affine_certificate = affine_certificate_factory(
        mock_match_result.source_points,
        mock_match_result.reference_points,
        np.array([True] * len(mock_match_result.source_points))
    )
    result = analyze_partial_correspondence(
        mock_match_result, mock_affine_certificate, image_shape, mock_verification_evidence, custom_config
    )

    # With stricter thresholds, we expect fewer supported regions
    assert result.status == PartialCorrespondenceStatus.INSUFFICIENT or result.status == PartialCorrespondenceStatus.PARTIAL
    assert len(result.supported_regions) < (custom_config.grid_shape[0] * custom_config.grid_shape[1])


def test_insufficient_data_regions(mock_match_result, affine_certificate_factory, mock_verification_evidence, default_partial_config):
    """Test that regions with no candidate matches are marked as insufficient data."""
    # Create a match result with points only in one corner, leaving other regions empty
    sparse_source_points = np.array([
        [10, 10], [20, 20], [30, 30], [40, 40],
    ], dtype=np.float32)
    sparse_reference_points = sparse_source_points + 1
    sparse_match_result = MatchResult(
        source_points=sparse_source_points,
        reference_points=sparse_reference_points,
        candidate_count=len(sparse_source_points),
        accepted_count=len(sparse_source_points),
        filter_diagnostics={},
    )
    sparse_inlier_mask = np.array([True] * len(sparse_source_points))
    sparse_affine_certificate = affine_certificate_factory(
        sparse_source_points,
        sparse_reference_points,
        sparse_inlier_mask
    )


    image_shape = (250, 250)
    result = analyze_partial_correspondence(
        sparse_match_result, sparse_affine_certificate, image_shape, mock_verification_evidence, default_partial_config
    )

    assert result.status == PartialCorrespondenceStatus.PARTIAL # One supported region, many insufficient
    assert len(result.insufficient_data_regions) > 0
    assert (0, 0) in result.supported_regions # Assuming (0,0) is where the points are
    assert (1, 0) in result.insufficient_data_regions # Example of an empty region
    assert result.local_failure_reasons[(1, 0)] == ["NO_CANDIDATE_MATCHES"]