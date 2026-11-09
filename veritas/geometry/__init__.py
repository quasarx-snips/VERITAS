"""AFFINE-ONLY geometric verification.

P0.4: migrated — split by responsibility:

- ``affine``      — Hartley-normalized 6-DoF affine solver, transformation
                    application, reprojection classification, and the
                    ``verify_affine`` certification entry point;
- ``ransac``      — the model-agnostic deterministic RANSAC engine;
- ``certificate`` — the structured geometric certificate (measurements only,
                    never probabilities).

Homography and automatic model selection from the reference repository are
explicitly excluded from VERITAS certification.
"""

from .affine import (
    MIN_AFFINE_SAMPLES,
    AffineVerificationConfig,
    AffineVerificationResult,
    apply_transformation,
    classify_matches,
    compute_error_statistics,
    compute_reprojection_errors,
    verify_affine,
    verify_affine_from_correspondences,
    verify_matches,
)
from .certificate import (
    AffineCertificate,
    ErrorStatistics,
    VerificationDiagnostics,
)
from .ransac import RansacConfig

__all__ = [
    "MIN_AFFINE_SAMPLES",
    "AffineCertificate",
    "AffineVerificationConfig",
    "AffineVerificationResult",
    "ErrorStatistics",
    "RansacConfig",
    "VerificationDiagnostics",
    "apply_transformation",
    "classify_matches",
    "compute_error_statistics",
    "compute_reprojection_errors",
    "verify_affine",
    "verify_affine_from_correspondences",
    "verify_matches",
]
