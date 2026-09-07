"""Robust selection and agreement checks for multi-detector geometry."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Tuple

import numpy as np

from ..geometry import AffineCertificate, apply_transformation
from ..spatial import CoverageConfig, calculate_spatial_coverage, calculate_spatial_entropy


def _quality(certificate: AffineCertificate, image_shape: Tuple[int, int], config: CoverageConfig | None) -> float:
    """Rank certified estimates by support quality, not detector preference."""
    if not certificate.is_valid:
        return float("-inf")
    coverage = calculate_spatial_coverage(certificate.source_inliers, image_shape, config)
    entropy = calculate_spatial_entropy(certificate.source_inliers, image_shape, config)
    # Count is deliberately log-capped: thousands of repeated texture matches
    # must not outweigh broad, geometrically clean support.
    count_score = min(np.log1p(certificate.inlier_count) / np.log(101.0), 1.0)
    residual_score = max(0.0, 1.0 - certificate.rmse / max(certificate.threshold or 3.0, 1e-6))
    return (
        0.35 * certificate.inlier_ratio
        + 0.25 * coverage["coverage_ratio"]
        + 0.15 * (entropy["normalized_entropy"] or 0.0)
        + 0.15 * count_score
        + 0.10 * residual_score
    )


def select_primary_certificate(
    certificates: Mapping[str, AffineCertificate],
    image_shape: Tuple[int, int],
    config: CoverageConfig | None = None,
) -> tuple[str, AffineCertificate]:
    """Choose the strongest *certified* detector estimate deterministically.

    The old pipeline always selected SIFT even when a second detector had much
    better distributed support.  Detector names only break exact score ties.
    """
    ranked = sorted(
        ((-_quality(certificate, image_shape, config), name, certificate) for name, certificate in certificates.items()),
        key=lambda item: (item[0], item[1]),
    )
    _, name, certificate = ranked[0]
    return name, certificate


def transform_agreement(
    certificates: Mapping[str, AffineCertificate],
    image_shape: Tuple[int, int],
    tolerance_fraction: float = 0.02,
) -> dict:
    """Compare certified detector transforms on image anchors.

    Matching detector states alone are insufficient: repeated patterns can make
    several detectors certify unrelated affine maps.  Agreement is measured in
    source-image-diagonal units so it remains stable across image resolutions.
    """
    h, w = image_shape[:2]
    anchors = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]], dtype=float)
    valid = {name: cert for name, cert in certificates.items() if cert.is_valid and cert.transformation is not None}
    diagonal = float(np.hypot(h, w))
    pairwise = {}
    agreeing = set()
    for left, right in combinations(sorted(valid), 2):
        delta = np.linalg.norm(
            apply_transformation(anchors, valid[left].transformation)
            - apply_transformation(anchors, valid[right].transformation),
            axis=1,
        )
        median_fraction = float(np.median(delta) / diagonal) if diagonal else float("inf")
        compatible = median_fraction <= tolerance_fraction
        pairwise[f"{left}:{right}"] = {
            "median_anchor_error_fraction": median_fraction,
            "compatible": compatible,
        }
        if compatible:
            agreeing.update((left, right))
    # One certified detector is evidence, but not cross-detector confirmation.
    return {
        "certified_detectors": sorted(valid),
        "pairwise": pairwise,
        "agreeing_detectors": sorted(agreeing),
        "agreement_strength": len(agreeing) / len(valid) if valid else 0.0,
        "tolerance_fraction": tolerance_fraction,
    }
