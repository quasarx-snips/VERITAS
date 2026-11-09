"""VERITAS geometry — structured geometric certificate (P0.4 migration).

A certificate records what the affine certification path measured. It is NOT a
probability and carries no confidence score: validity, inlier/outlier counts,
reprojection residual statistics and the estimated transformation only.

All error magnitudes are pixel-space reprojection residuals (image-based
correspondence metrics). They are not geographic accuracies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ErrorStatistics:
    """Reprojection error statistics for inlier matches (pixel units)."""

    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    rmse: float = 0.0
    min_error: float = 0.0
    max_error: float = 0.0
    percentile_95: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VerificationDiagnostics:
    """Detailed diagnostics resulting from geometric verification."""

    model_name: str
    total_matches: int
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    iterations_run: int
    converged: bool
    is_valid: bool
    error_stats: ErrorStatistics
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["error_stats"] = self.error_stats.to_dict()
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"[{status}] Model: {self.model_name} | "
            f"Inliers: {self.inlier_count}/{self.total_matches} "
            f"({self.inlier_ratio * 100:.1f}%) | "
            f"RMSE: {self.error_stats.rmse:.3f}px | "
            f"Iterations: {self.iterations_run}"
        )


@dataclass
class AffineCertificate:
    """Geometric certificate issued by the AFFINE-ONLY certification path.

    ``model`` is always ``"affine"`` — the certification path accepts no other
    model and records the exact deterministic configuration used.
    """

    is_valid: bool
    transformation: Optional[np.ndarray]      # 3x3 affine matrix (source -> reference)
    inlier_mask: np.ndarray                   # 1D boolean array (True = inlier)
    diagnostics: VerificationDiagnostics
    source_inliers: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    reference_inliers: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    inlier_indices: Optional[np.ndarray] = None
    configuration: Dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> str:
        """Certification model name — always ``"affine"`` in VERITAS."""
        return self.diagnostics.model_name

    @property
    def candidate_count(self) -> int:
        return int(self.diagnostics.total_matches)

    @property
    def inlier_count(self) -> int:
        return int(self.diagnostics.inlier_count)

    @property
    def outlier_count(self) -> int:
        return int(self.diagnostics.outlier_count)

    @property
    def inlier_ratio(self) -> float:
        return float(self.diagnostics.inlier_ratio)

    @property
    def rmse(self) -> float:
        return float(self.diagnostics.error_stats.rmse)

    @property
    def median_error(self) -> float:
        return float(self.diagnostics.error_stats.median)

    @property
    def p95_error(self) -> float:
        return float(self.diagnostics.error_stats.percentile_95)

    @property
    def max_error(self) -> float:
        return float(self.diagnostics.error_stats.max_error)

    @property
    def threshold(self) -> Optional[float]:
        value = self.configuration.get("reprojection_threshold")
        if value is None:
            value = self.diagnostics.additional_info.get("reprojection_threshold")
        return None if value is None else float(value)

    @property
    def minimum_inliers(self) -> Optional[int]:
        value = self.configuration.get("min_inliers")
        return None if value is None else int(value)

    def get_transformation_2x3(self) -> Optional[np.ndarray]:
        """Return transformation as a 2x3 matrix for OpenCV warp functions."""
        if self.transformation is None:
            return None
        if self.transformation.shape == (2, 3):
            return self.transformation.copy()
        if self.transformation.shape == (3, 3):
            return self.transformation[:2, :].copy()
        raise ValueError(f"Unexpected transformation shape: {self.transformation.shape}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        return {
            "is_valid": self.is_valid,
            "model": self.model,
            "transformation": self.transformation.tolist() if self.transformation is not None else None,
            "candidate_count": self.candidate_count,
            "inlier_count": self.inlier_count,
            "outlier_count": self.outlier_count,
            "inlier_ratio": self.inlier_ratio,
            "rmse_px": self.rmse,
            "median_error_px": self.median_error,
            "p95_error_px": self.p95_error,
            "max_error_px": self.max_error,
            "threshold_px": self.threshold,
            "minimum_inliers": self.minimum_inliers,
            "configuration": dict(self.configuration),
        }


#: Backwards-compatible alias for the pre-split name.
AffineVerificationResult = AffineCertificate
