"""Unit tests for the VERITAS verdict classifier."""

import pytest
from unittest.mock import Mock

from veritas.schemas import (
    CounterEvidence, DetectorEvidence, GeometryEvidence, MatchEvidence,
    QuorumEvidence, SpatialEvidence, VerificationEvidence
)
from veritas.verdict.classifier import VerdictClassifier
from veritas.verdict.schema import Verdict, VerdictResult
from veritas.verdict.thresholds import VerdictThresholds


@pytest.fixture
def default_thresholds():
    """Fixture for default VerdictThresholds."""
    return VerdictThresholds()


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

    mock_features = {"sift": Mock(spec=DetectorEvidence)}
    mock_matches = {"sift": Mock(spec=MatchEvidence)}

    evidence = Mock(spec=VerificationEvidence)
    evidence.geometry = mock_geometry
    evidence.spatial = mock_spatial
    evidence.quorum = mock_quorum
    evidence.counter_evidence = mock_counter_evidence
    evidence.features = mock_features
    evidence.matches = mock_matches
    evidence.evidence_score = 1.0
    evidence.to_dict.return_value = {}
    return evidence


def test_strong_verdict(default_thresholds, mock_verification_evidence):
    """Test case for strong evidence leading to a STRONG verdict."""
    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.STRONG
    assert "AFFINE_CERTIFIED" in result.rationale_codes
    assert "DISTRIBUTED_SUPPORT" in result.rationale_codes
    assert "DETECTOR_AGREEMENT" in result.rationale_codes
    assert "FULL_CORRESPONDENCE_SUPPORT" in result.rationale_codes


def test_affine_failure_verdict(default_thresholds, mock_verification_evidence):
    """Test case for affine failure leading to a NONE verdict."""
    mock_verification_evidence.geometry.certified = False
    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.NONE
    assert "AFFINE_FAILED" in result.rationale_codes
    assert "AFFINE_CERTIFIED" not in result.rationale_codes


def test_partial_verdict(default_thresholds, mock_verification_evidence):
    """Test case for partial evidence leading to a PARTIAL verdict."""
    mock_verification_evidence.geometry.inlier_ratio = 0.5
    mock_verification_evidence.geometry.inlier_count = 30
    mock_verification_evidence.spatial.coverage_ratio = 0.4
    mock_verification_evidence.spatial.normalized_entropy = 0.5
    mock_verification_evidence.quorum.quorum_strength = 0.6
    mock_verification_evidence.counter_evidence.residuals["max_reprojection_error"] = 3.0
    mock_verification_evidence.counter_evidence.residuals["mean_reprojection_error"] = 1.5

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "PARTIAL")

    assert result.verdict == Verdict.PARTIAL
    assert "PARTIAL_OVERLAP" in result.rationale_codes
    assert "PARTIAL_CORRESPONDENCE_SUPPORT" in result.rationale_codes
    assert "DISTRIBUTED_SUPPORT" not in result.rationale_codes


def test_weak_verdict(default_thresholds, mock_verification_evidence):
    """Test case for weak evidence leading to a WEAK verdict."""
    mock_verification_evidence.geometry.inlier_ratio = 0.15
    mock_verification_evidence.geometry.inlier_count = 10
    mock_verification_evidence.spatial.coverage_ratio = 0.1
    mock_verification_evidence.spatial.normalized_entropy = 0.1
    mock_verification_evidence.quorum.quorum_strength = 0.2
    mock_verification_evidence.counter_evidence.residuals["max_reprojection_error"] = 8.0
    mock_verification_evidence.counter_evidence.residuals["mean_reprojection_error"] = 4.0

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.WEAK
    assert "INSUFFICIENT_CORRESPONDENCE" in result.rationale_codes
    assert "PARTIAL_OVERLAP" not in result.rationale_codes


def test_none_verdict_insufficient_inliers(default_thresholds, mock_verification_evidence):
    """Test case for insufficient inliers leading to a NONE verdict."""
    mock_verification_evidence.geometry.inlier_count = 2
    mock_verification_evidence.geometry.inlier_ratio = 0.05
    mock_verification_evidence.spatial.coverage_ratio = 0.01
    mock_verification_evidence.spatial.normalized_entropy = 0.01

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "INSUFFICIENT")

    assert result.verdict == Verdict.NONE
    assert "INSUFFICIENT_CORRESPONDENCE" in result.rationale_codes


def test_high_residual_counter_evidence(default_thresholds, mock_verification_evidence):
    """Test case for high residual counter-evidence."""
    mock_verification_evidence.counter_evidence.residuals["max_reprojection_error"] = 10.0
    mock_verification_evidence.counter_evidence.residuals["mean_reprojection_error"] = 5.0

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.PARTIAL  # Assuming other strong conditions are met, but residuals push it to partial
    assert "HIGH_RESIDUAL" in result.rationale_codes


def test_detector_disagreement_affects_evidence(default_thresholds, mock_verification_evidence):
    """Test case for detector disagreement affecting the verdict."""
    mock_verification_evidence.counter_evidence.feature_disagreement["disagreement_score"] = 0.8
    mock_verification_evidence.geometry.inlier_ratio = 0.2
    mock_verification_evidence.geometry.inlier_count = 10

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.WEAK  # Disagreement pushes it down
    assert "DETECTOR_DISAGREEMENT" in result.rationale_codes


def test_low_spatial_coverage(default_thresholds, mock_verification_evidence):
    """Test case for low spatial coverage."""
    mock_verification_evidence.spatial.coverage_ratio = 0.1

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "INSUFFICIENT")

    assert result.verdict == Verdict.WEAK
    assert "LOW_SPATIAL_COVERAGE" in result.rationale_codes


def test_low_spatial_entropy(default_thresholds, mock_verification_evidence):
    """Test case for low spatial entropy."""
    mock_verification_evidence.spatial.normalized_entropy = 0.1

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "INSUFFICIENT")

    assert result.verdict == Verdict.WEAK
    assert "LOW_SPATIAL_ENTROPY" in result.rationale_codes


def test_high_spatial_concentration(default_thresholds, mock_verification_evidence):
    """Test case for high spatial concentration."""
    mock_verification_evidence.counter_evidence.spatial_concentration["concentration_score"] = 0.8

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.PARTIAL
    assert "HIGH_SPATIAL_CONCENTRATION" in result.rationale_codes


def test_heuristic_score_not_labeled_probability(default_thresholds, mock_verification_evidence):
    """Test that heuristic scores are not labeled as probabilities."""
    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    for key in result.confidence_status:
        assert "probability" not in key.lower()


def test_missing_evidence_does_not_crash(default_thresholds, mock_verification_evidence):
    """Test that missing optional evidence does not crash the classifier."""
    mock_verification_evidence.spatial.normalized_entropy = None
    mock_verification_evidence.counter_evidence.residuals = {}
    mock_verification_evidence.counter_evidence.feature_disagreement = {}
    mock_verification_evidence.counter_evidence.spatial_concentration = {}

    classifier = VerdictClassifier(default_thresholds)
    result = classifier.classify(mock_verification_evidence, "FULL")

    assert result.verdict == Verdict.STRONG  # Should still be strong if other conditions met
    assert "LOW_SPATIAL_ENTROPY" not in result.rationale_codes
    assert "HIGH_RESIDUAL" not in result.rationale_codes
    assert "DETECTOR_DISAGREEMENT" not in result.rationale_codes
    assert "HIGH_SPATIAL_CONCENTRATION" not in result.rationale_codes