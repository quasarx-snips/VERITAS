"""AFFINE-ONLY geometric verification.

P0.4: migrated — deterministic vectorized RANSAC, Hartley-normalized affine
solver, collinearity guard, reprojection error classification, and verification
diagnostics (``verify_affine``). Homography and automatic model selection from
the reference repository are explicitly excluded from VERITAS certification.
"""

from .affine import (
    MIN_AFFINE_SAMPLES,
    AffineVerificationConfig,
    AffineVerificationResult,
    ErrorStatistics,
    VerificationDiagnostics,
    apply_transformation,
    classify_matches,
    compute_error_statistics,
    compute_reprojection_errors,
    verify_affine,
    verify_affine_from_correspondences,
)

__all__ = [
    "MIN_AFFINE_SAMPLES",
    "AffineVerificationConfig",
    "AffineVerificationResult",
    "ErrorStatistics",
    "VerificationDiagnostics",
    "apply_transformation",
    "classify_matches",
    "compute_error_statistics",
    "compute_reprojection_errors",
    "verify_affine",
    "verify_affine_from_correspondences",
]