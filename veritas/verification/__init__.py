"""Verification metrics and evidence fusion primitives.

P0.5: migrated — inlier-ratio, reprojection-error statistics, RMSE, and
registration evaluation (``evaluate_registration``) from the reference metrics
module; spatial coverage lives in ``veritas.spatial``. The fused evidence score
that uses these metrics arrives with the P1 verdict engine and is a heuristic
evidence score, never presented as a probability or calibrated confidence.
"""

from .metrics import (
    EvaluationThresholds,
    calculate_error_statistics,
    calculate_inlier_ratio,
    calculate_reprojection_errors,
    calculate_rmse,
    evaluate_registration,
)
from .evidence import geometry_evidence
from .fusion import fuse_evidence
from .selection import select_primary_certificate, transform_agreement

__all__ = [
    "EvaluationThresholds",
    "calculate_error_statistics",
    "calculate_inlier_ratio",
    "calculate_reprojection_errors",
    "calculate_rmse",
    "evaluate_registration",
    "geometry_evidence",
    "fuse_evidence",
    "select_primary_certificate",
    "transform_agreement",
]
