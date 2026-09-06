"""Phase 1 image-pair verification orchestration.

The pipeline intentionally ends at raw affine and spatial evidence. It does
not fuse evidence, make a verdict, or invoke downstream gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .features import SiftDetector
from .geometry import AffineCertificate, AffineVerificationConfig, verify_affine_from_correspondences
from .matching import MatchResult, match_feature_sets
from .preprocessing import ImagePreprocessor, PreprocessedImage
from .spatial import CoverageConfig, calculate_spatial_coverage
from .verification import evaluate_registration


@dataclass
class Phase1VerificationResult:
    """Raw Phase 1 evidence produced for one source/reference image pair."""

    source: PreprocessedImage
    reference: PreprocessedImage
    source_feature_count: int
    reference_feature_count: int
    matches: MatchResult
    certificate: AffineCertificate
    metrics: Optional[Dict[str, Any]]
    spatial_coverage: Dict[str, Any]


def verify_image_pair(
    source_image: Any,
    reference_image: Any,
    *,
    preprocessor: Optional[ImagePreprocessor] = None,
    detector: Optional[SiftDetector] = None,
    matching_config: Optional[Dict[str, Any]] = None,
    geometry_config: Optional[AffineVerificationConfig] = None,
    coverage_config: Optional[CoverageConfig] = None,
) -> Phase1VerificationResult:
    """Run preprocessing through AFFINE-only spatial evidence.

    ``source_image`` coordinates are transformed into ``reference_image``
    coordinates. An invalid affine certificate remains raw evidence rather
    than being converted into a verdict or confidence score.
    """
    image_preprocessor = preprocessor or ImagePreprocessor()
    sift = detector or SiftDetector()
    source = image_preprocessor.process(source_image)
    reference = image_preprocessor.process(reference_image)
    source_features, source_descriptors = sift.detect(source.enhanced)
    reference_features, reference_descriptors = sift.detect(reference.enhanced)
    matches = match_feature_sets(
        (source_features, source_descriptors),
        (reference_features, reference_descriptors),
        matching_config,
    )
    certificate = verify_affine_from_correspondences(matches, geometry_config)
    coverage = calculate_spatial_coverage(
        certificate.source_inliers,
        source.enhanced.shape,
        coverage_config,
    )
    metrics = None
    if certificate.transformation is not None:
        metrics = evaluate_registration(
            matches.source_points,
            matches.reference_points,
            certificate.transformation,
            certificate.inlier_mask,
            image_shape=source.enhanced.shape,
        )
    return Phase1VerificationResult(
        source=source,
        reference=reference,
        source_feature_count=len(source_features),
        reference_feature_count=len(reference_features),
        matches=matches,
        certificate=certificate,
        metrics=metrics,
        spatial_coverage=coverage,
    )
