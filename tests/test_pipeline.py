"""End-to-end Phase 1 affine smoke test."""

import cv2
import numpy as np

from veritas.geometry import AffineVerificationConfig
from veritas.pipeline import verify_image_pair


def _textured_image() -> np.ndarray:
    image = np.zeros((300, 300), dtype=np.uint8)
    for x, y, radius, intensity in ((55, 70, 19, 255), (180, 65, 31, 180), (95, 210, 25, 220)):
        cv2.circle(image, (x, y), radius, intensity, 3)
    cv2.rectangle(image, (175, 165), (260, 245), 200, 3)
    cv2.line(image, (20, 270), (275, 35), 240, 3)
    cv2.putText(image, "V", (125, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3)
    return image


def test_phase1_pipeline_smoke_produces_affine_certificate_and_metrics():
    source = _textured_image()
    transform = np.float32([[1.02, 0.04, 9.0], [-0.03, 0.98, -7.0]])
    reference = cv2.warpAffine(source, transform, (300, 300))

    result = verify_image_pair(
        source,
        reference,
        matching_config={"method": "FLANN", "ratio": 0.8, "mutual_consistency": True},
        geometry_config=AffineVerificationConfig(reprojection_threshold=2.0, min_inliers=6, random_seed=17),
    )

    assert result.source_feature_count > 0
    assert result.reference_feature_count > 0
    assert result.matches.accepted_count >= 6
    assert result.certificate.is_valid
    assert result.certificate.model == "affine"
    assert np.isfinite(result.certificate.rmse)
    assert result.metrics is not None
    assert np.isfinite(result.metrics["inlier_error_statistics"]["rmse"])
    assert np.isfinite(result.spatial_coverage["coverage_ratio"])
