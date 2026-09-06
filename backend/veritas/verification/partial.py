"""Partial correspondence analysis for VERITAS.

This module implements the partial correspondence analysis, which assesses
the support for correspondence within defined spatial regions (grid cells).
It does not produce a global VERITAS verdict but rather provides structured
evidence to the verdict classifier.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from veritas.geometry.certificate import AffineCertificate
from veritas.matching.descriptor_matching import MatchResult
from veritas.schemas import VerificationEvidence
from veritas.spatial.coverage import CoverageConfig, calculate_spatial_coverage


class PartialCorrespondenceStatus(str, Enum):
    """Status of partial correspondence analysis."""

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class RegionEvidence:
    """Evidence for a single grid region."""

    row: int
    col: int
    status: str  # "SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_DATA"
    inlier_count: int
    candidate_match_count: int
    inlier_ratio: float
    mean_residual: Optional[float] = None
    max_residual: Optional[float] = None
    detector_support_score: Optional[float] = None
    detector_disagreement_score: Optional[float] = None
    local_failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartialCorrespondenceResult:
    """Result of the partial correspondence analysis."""

    status: PartialCorrespondenceStatus
    supported_regions: List[Tuple[int, int]] = field(default_factory=list)
    unsupported_regions: List[Tuple[int, int]] = field(default_factory=list)
    insufficient_data_regions: List[Tuple[int, int]] = field(default_factory=list)
    coverage_fraction: float = 0.0
    region_evidence: List[RegionEvidence] = field(default_factory=list)
    local_failure_reasons: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "supported_regions": self.supported_regions,
            "unsupported_regions": self.unsupported_regions,
            "insufficient_data_regions": self.insufficient_data_regions,
            "coverage_fraction": self.coverage_fraction,
            "region_evidence": [re.to_dict() for re in self.region_evidence],
            "local_failure_reasons": {str(k): v for k, v in self.local_failure_reasons.items()},
        }


@dataclass(frozen=True)
class PartialCorrespondenceConfig:
    """Configuration for partial correspondence analysis."""

    grid_shape: Tuple[int, int] = (4, 4)
    min_inliers_per_region: int = 3
    min_inlier_ratio_per_region: float = 0.3
    max_mean_residual_per_region: float = 3.0
    max_detector_disagreement_per_region: float = 0.5


def analyze_partial_correspondence(
    match_result: MatchResult,
    certificate: AffineCertificate,
    image_shape: Tuple[int, int],
    verification_evidence: VerificationEvidence,
    config: Optional[PartialCorrespondenceConfig] = None,
) -> PartialCorrespondenceResult:
    """Analyzes correspondence support across spatial regions.

    Args:
        match_result: The raw match results before RANSAC.
        certificate: The affine certificate containing inlier masks and transformation.
        image_shape: The shape of the source image (height, width).
        verification_evidence: The full verification evidence bundle.
        config: Configuration for partial correspondence analysis.

    Returns:
        A PartialCorrespondenceResult detailing regional support.
    """
    cfg = config or PartialCorrespondenceConfig()
    h, w = image_shape
    rows, cols = cfg.grid_shape

    # Prepare data for grid analysis
    source_points = match_result.source_points
    inlier_mask = certificate.inlier_mask
    all_matches_indices = np.arange(len(source_points))

    # Grid cell dimensions
    cell_width = w / cols
    cell_height = h / rows

    all_region_evidence: List[RegionEvidence] = []
    supported_regions: List[Tuple[int, int]] = []
    unsupported_regions: List[Tuple[int, int]] = []
    insufficient_data_regions: List[Tuple[int, int]] = []
    local_failure_reasons: Dict[Tuple[int, int], List[str]] = {}

    for r in range(rows):
        for c in range(cols):
            region_failure_reasons: List[str] = []
            region_status = "UNSUPPORTED"  # Default status

            # Define region boundaries
            x_min, x_max = c * cell_width, (c + 1) * cell_width
            y_min, y_max = r * cell_height, (r + 1) * cell_height

            # Find matches within this region
            points_in_region_mask = (
                (source_points[:, 0] >= x_min) & (source_points[:, 0] < x_max) &
                (source_points[:, 1] >= y_min) & (source_points[:, 1] < y_max)
            )
            candidate_matches_in_region_indices = all_matches_indices[points_in_region_mask]
            candidate_match_count = len(candidate_matches_in_region_indices)

            if candidate_match_count == 0:
                region_status = "INSUFFICIENT_DATA"
                insufficient_data_regions.append((r, c))
                region_failure_reasons.append("NO_CANDIDATE_MATCHES")
                all_region_evidence.append(
                    RegionEvidence(r, c, region_status, 0, 0, 0.0, local_failure_reasons=region_failure_reasons)
                )
                local_failure_reasons[(r, c)] = region_failure_reasons
                continue

            # Determine inliers within this region
            inliers_in_region_mask = inlier_mask[candidate_matches_in_region_indices]
            inlier_count = np.sum(inliers_in_region_mask)
            inlier_ratio = inlier_count / candidate_match_count if candidate_match_count > 0 else 0.0

            # Calculate residuals for inliers in this region
            mean_residual, max_residual = None, None
            if inlier_count > 0 and certificate.transformation is not None:
                source_inliers_in_region = source_points[candidate_matches_in_region_indices][inliers_in_region_mask]
                reference_inliers_in_region = match_result.reference_points[candidate_matches_in_region_indices][inliers_in_region_mask]
                if len(source_inliers_in_region) > 0:
                    from veritas.geometry.affine import compute_reprojection_errors
                    residuals = compute_reprojection_errors(
                        source_inliers_in_region,
                        reference_inliers_in_region,
                        certificate.transformation
                    )
                    mean_residual = np.mean(residuals)
                    max_residual = np.max(residuals)

            # Assess detector support/disagreement (simplified for regional)
            # For now, we'll use global scores as a proxy, or assume uniform distribution
            # A more sophisticated approach would re-evaluate quorum per region.
            detector_support_score = verification_evidence.quorum.quorum_strength
            detector_disagreement_score = verification_evidence.counter_evidence.feature_disagreement.get("disagreement_score", 0.0)

            # Apply regional support criteria
            if (inlier_count >= cfg.min_inliers_per_region and
                    inlier_ratio >= cfg.min_inlier_ratio_per_region and
                    (mean_residual is None or mean_residual <= cfg.max_mean_residual_per_region) and
                    detector_disagreement_score <= cfg.max_detector_disagreement_per_region):
                region_status = "SUPPORTED"
                supported_regions.append((r, c))
            else:
                unsupported_regions.append((r, c))
                if inlier_count < cfg.min_inliers_per_region:
                    region_failure_reasons.append("INSUFFICIENT_INLIERS")
                if inlier_ratio < cfg.min_inlier_ratio_per_region:
                    region_failure_reasons.append("LOW_INLIER_RATIO")
                if mean_residual is not None and mean_residual > cfg.max_mean_residual_per_region:
                    region_failure_reasons.append("HIGH_RESIDUAL")
                if detector_disagreement_score > cfg.max_detector_disagreement_per_region:
                    region_failure_reasons.append("HIGH_DETECTOR_DISAGREEMENT")

            all_region_evidence.append(
                RegionEvidence(
                    r, c, region_status, inlier_count, candidate_match_count, inlier_ratio,
                    mean_residual, max_residual, detector_support_score, detector_disagreement_score,
                    local_failure_reasons=region_failure_reasons
                )
            )
            if region_failure_reasons:
                local_failure_reasons[(r, c)] = region_failure_reasons

    total_cells = rows * cols
    supported_fraction = len(supported_regions) / total_cells if total_cells > 0 else 0.0

    overall_status: PartialCorrespondenceStatus
    if len(supported_regions) == total_cells:
        overall_status = PartialCorrespondenceStatus.FULL
    elif len(supported_regions) > 0:
        overall_status = PartialCorrespondenceStatus.PARTIAL
    else:
        overall_status = PartialCorrespondenceStatus.INSUFFICIENT

    return PartialCorrespondenceResult(
        status=overall_status,
        supported_regions=supported_regions,
        unsupported_regions=unsupported_regions,
        insufficient_data_regions=insufficient_data_regions,
        coverage_fraction=supported_fraction,
        region_evidence=all_region_evidence,
        local_failure_reasons=local_failure_reasons,
    )