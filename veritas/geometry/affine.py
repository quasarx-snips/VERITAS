"""VERITAS geometry — AFFINE-ONLY geometric verification (P0.4 migration).

Migrated from the reference repository's geometric verification module. The
proven behavior is preserved:

- Hartley isotropic point normalisation for the least-squares affine solver;
- deterministic, vectorized RANSAC with an adaptive iteration bound and a
  refit-on-inliers pass;
- collinearity guard for minimal samples;
- reprojection error classification and error statistics.

VERITAS certification policy (see ``veritas.config``):
- The affine model is the ONLY accepted certification model.
- No automatic model selection is performed (the reference's "auto" ranking is
  intentionally not migrated).
- Homography is NOT an accepted certification model (the reference's projective
  solver is reference-only material, not migrated here).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

#: Minimal sample size for the affine model (6 DoF require 3 point pairs).
MIN_AFFINE_SAMPLES: int = 3


@dataclass
class AffineVerificationConfig:
    """Configuration parameters for affine geometric verification."""

    reprojection_threshold: float = 3.0      # Max pixel distance for inliers
    confidence: float = 0.99                 # RANSAC desired confidence level
    max_iterations: int = 2000               # Maximum RANSAC iterations
    min_inliers: int = 4                     # Minimum required inliers for success
    refit_inliers: bool = True               # Refit model on all inliers via LSQ
    random_seed: Optional[int] = 42          # RNG seed for deterministic execution


@dataclass
class ErrorStatistics:
    """Reprojection error statistics for inlier matches."""

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
class AffineVerificationResult:
    """Complete output from affine geometric verification."""

    is_valid: bool
    transformation: Optional[np.ndarray]      # 3x3 affine matrix (source -> reference)
    inlier_mask: np.ndarray                   # 1D boolean array (True = inlier)
    diagnostics: VerificationDiagnostics
    source_inliers: np.ndarray                # (K, 2) inlier points in source
    reference_inliers: np.ndarray             # (K, 2) inlier points in reference
    inlier_indices: Optional[np.ndarray] = None

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
            "transformation": self.transformation.tolist() if self.transformation is not None else None,
            "inlier_count": int(self.diagnostics.inlier_count),
            "inlier_ratio": float(self.diagnostics.inlier_ratio),
            "rmse": float(self.diagnostics.error_stats.rmse),
            "model": self.diagnostics.model_name,
        }


# =====================================================================
# Vectorized affine solvers (direct linear estimation)
# =====================================================================

def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization for numerical stability."""
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2.0) / (mean_dist + 1e-12)

    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pts_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    normalized_pts = (T @ pts_h.T).T[:, :2]
    return normalized_pts, T


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Vectorized 2D Affine Transformation Solver (6 DoF)."""
    num_pts = len(src)
    if num_pts < MIN_AFFINE_SAMPLES:
        return None

    src_norm, T_src = _normalize_points(src)
    dst_norm, T_dst = _normalize_points(dst)

    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]

    A = np.zeros((2 * num_pts, 6), dtype=np.float64)
    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1.0

    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1.0

    b = np.empty(2 * num_pts, dtype=np.float64)
    b[0::2] = u
    b[1::2] = v

    try:
        sol, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 6:
            return None
        M_norm = np.array(
            [
                [sol[0], sol[1], sol[2]],
                [sol[3], sol[4], sol[5]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        T = np.linalg.inv(T_dst) @ M_norm @ T_src
        return T / (T[2, 2] + 1e-12)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _are_points_collinear(points: np.ndarray, eps: float = 1e-6) -> bool:
    """Constant-time vector cross-product collinearity check for minimal samples."""
    n = len(points)
    if n < 3:
        return False

    if n == 3:
        p0, p1, p2 = points[0], points[1], points[2]
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        return area < eps
    elif n == 4:
        v01 = points[1] - points[0]
        v02 = points[2] - points[0]
        v03 = points[3] - points[0]
        if abs(v01[0] * v02[1] - v01[1] * v02[0]) < eps:
            return True
        if abs(v01[0] * v03[1] - v01[1] * v03[0]) < eps:
            return True
        if abs(v02[0] * v03[1] - v02[1] * v03[0]) < eps:
            return True
        v12 = points[2] - points[1]
        v13 = points[3] - points[1]
        if abs(v12[0] * v13[1] - v12[1] * v13[0]) < eps:
            return True
        return False

    for i in range(n - 2):
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        if area < eps:
            return True
    return False


def apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Apply a 3x3 linear transform to (N, 2) points without allocation churn."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points.shape}")

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    if transformation.shape == (2, 3):
        transformation = np.vstack([transformation, [0.0, 0.0, 1.0]])

    x, y = points[:, 0], points[:, 1]
    u = transformation[0, 0] * x + transformation[0, 1] * y + transformation[0, 2]
    v = transformation[1, 0] * x + transformation[1, 1] * y + transformation[1, 2]
    w = transformation[2, 0] * x + transformation[2, 1] * y + transformation[2, 2]

    w = np.where(np.abs(w) < 1e-12, 1e-12, w)

    projected = np.empty_like(points)
    projected[:, 0] = u / w
    projected[:, 1] = v / w
    return projected


def compute_reprojection_errors(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transformation: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean reprojection distance in single-pass NumPy calls."""
    projected = apply_transformation(source_points, transformation)
    return np.hypot(reference_points[:, 0] - projected[:, 0], reference_points[:, 1] - projected[:, 1])


def compute_error_statistics(errors: np.ndarray) -> ErrorStatistics:
    """Calculate summary statistics on an array of errors."""
    if len(errors) == 0:
        return ErrorStatistics()

    return ErrorStatistics(
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        min_error=float(np.min(errors)),
        max_error=float(np.max(errors)),
        percentile_95=float(np.percentile(errors, 95)),
    )


def classify_matches(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transformation: Optional[np.ndarray],
    threshold: float,
) -> np.ndarray:
    """Classify correspondences using fast distance thresholding."""
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(reference_points, dtype=np.float64)

    if len(src) != len(dst):
        raise ValueError(f"Point counts mismatch: {len(src)} vs {len(dst)}")

    if len(src) == 0 or transformation is None:
        return np.zeros(len(src), dtype=bool)

    errors = compute_reprojection_errors(src, dst, transformation)
    return (errors <= threshold) & np.isfinite(errors)


def _config(config: Optional[Any]) -> AffineVerificationConfig:
    """Coerce a config value: None / AffineVerificationConfig / dict."""
    if config is None:
        return AffineVerificationConfig()
    if isinstance(config, AffineVerificationConfig):
        return config
    if isinstance(config, dict):
        return AffineVerificationConfig(**config)
    raise TypeError("config must be None, AffineVerificationConfig, or a dictionary")


def _ransac_affine(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    config: AffineVerificationConfig,
) -> Tuple[Optional[np.ndarray], np.ndarray, int, bool]:
    """Internal deterministic RANSAC engine for the affine model."""
    num_pts = len(source_points)
    if num_pts < MIN_AFFINE_SAMPLES:
        return None, np.zeros(num_pts, dtype=bool), 0, False

    rng = np.random.RandomState(config.random_seed)

    best_inlier_mask = np.zeros(num_pts, dtype=bool)
    best_inlier_count = 0
    best_model = None
    dynamic_max_iters = config.max_iterations
    iterations_run = 0

    for it in range(config.max_iterations):
        iterations_run += 1
        if iterations_run > dynamic_max_iters:
            break

        sample_indices = rng.choice(num_pts, size=MIN_AFFINE_SAMPLES, replace=False)
        src_sample = source_points[sample_indices]
        dst_sample = reference_points[sample_indices]

        if _are_points_collinear(src_sample):
            continue

        model_candidate = _fit_affine(src_sample, dst_sample)
        if model_candidate is None or not np.all(np.isfinite(model_candidate)):
            continue

        inlier_mask = classify_matches(
            source_points, reference_points, model_candidate, config.reprojection_threshold
        )
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_model = model_candidate

            w = inlier_count / float(num_pts)
            p = config.confidence
            w_sample = max(w ** MIN_AFFINE_SAMPLES, 1e-12)
            if 1.0 - w_sample > 0.0:
                calc_iters = math.log(1.0 - p) / math.log(1.0 - w_sample)
                dynamic_max_iters = min(config.max_iterations, int(math.ceil(calc_iters)))

    if config.refit_inliers and best_inlier_count >= MIN_AFFINE_SAMPLES:
        inlier_src = source_points[best_inlier_mask]
        inlier_dst = reference_points[best_inlier_mask]
        refit_model = _fit_affine(inlier_src, inlier_dst)
        if refit_model is not None and np.all(np.isfinite(refit_model)):
            best_model = refit_model
            best_inlier_mask = classify_matches(
                source_points, reference_points, best_model, config.reprojection_threshold
            )
            best_inlier_count = int(np.sum(best_inlier_mask))

    is_converged = best_inlier_count >= config.min_inliers
    return best_model, best_inlier_mask, iterations_run, is_converged


def verify_affine(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    config: Optional[Any] = None,
) -> AffineVerificationResult:
    """Estimate an affine certificate for a set of point correspondences.

    The model is always affine: no automatic model selection is performed and
    no other model is accepted (AFFINE-ONLY certification policy).
    """
    cfg = _config(config)
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(reference_points, dtype=np.float64)
    total_pts = len(src)

    if total_pts != len(dst):
        raise ValueError(f"Mismatch in point sizes: {len(src)} vs {len(dst)}")

    if total_pts < MIN_AFFINE_SAMPLES:
        diag = VerificationDiagnostics(
            model_name="affine",
            total_matches=total_pts,
            inlier_count=0,
            outlier_count=total_pts,
            inlier_ratio=0.0,
            iterations_run=0,
            converged=False,
            is_valid=False,
            error_stats=ErrorStatistics(),
            additional_info={"reason": "Insufficient points for affine verification"},
        )
        return AffineVerificationResult(
            is_valid=False,
            transformation=None,
            inlier_mask=np.zeros(total_pts, dtype=bool),
            diagnostics=diag,
            source_inliers=np.empty((0, 2), dtype=np.float64),
            reference_inliers=np.empty((0, 2), dtype=np.float64),
            inlier_indices=None,
        )

    transformation, inlier_mask, iterations_run, converged = _ransac_affine(src, dst, cfg)
    inlier_count = int(np.sum(inlier_mask))

    if transformation is not None and inlier_count > 0:
        inlier_errors = compute_reprojection_errors(src[inlier_mask], dst[inlier_mask], transformation)
        error_stats = compute_error_statistics(inlier_errors)
    else:
        error_stats = ErrorStatistics()

    is_valid = bool(converged and inlier_count >= cfg.min_inliers and transformation is not None)
    ratio = float(inlier_count / total_pts) if total_pts > 0 else 0.0

    diagnostics = VerificationDiagnostics(
        model_name="affine",
        total_matches=total_pts,
        inlier_count=inlier_count,
        outlier_count=total_pts - inlier_count,
        inlier_ratio=ratio,
        iterations_run=iterations_run,
        converged=converged,
        is_valid=is_valid,
        error_stats=error_stats,
        additional_info={"reprojection_threshold": cfg.reprojection_threshold},
    )

    return AffineVerificationResult(
        is_valid=is_valid,
        transformation=transformation if is_valid else None,
        inlier_mask=inlier_mask,
        diagnostics=diagnostics,
        source_inliers=src[inlier_mask] if is_valid else np.empty((0, 2), dtype=np.float64),
        reference_inliers=dst[inlier_mask] if is_valid else np.empty((0, 2), dtype=np.float64),
        inlier_indices=np.where(inlier_mask)[0] if is_valid else None,
    )


def verify_affine_from_correspondences(
    match_result: Any,
    config: Optional[Any] = None,
) -> AffineVerificationResult:
    """Convenience wrapper that accepts a matcher ``MatchResult`` directly."""
    from ..matching import MatchResult

    if not isinstance(match_result, MatchResult):
        raise TypeError("match_result must be a veritas.matching.MatchResult")
    return verify_affine(match_result.source_points, match_result.reference_points, config=config)