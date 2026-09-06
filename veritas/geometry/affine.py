"""VERITAS geometry — AFFINE-ONLY estimation (P0.4 migration).

Responsibility split:

- ``affine``      — the 6-DoF affine model: Hartley point normalisation, the
                    least-squares solver, application to points, reprojection
                    error classification and the ``verify_affine`` entry point;
- ``ransac``      — the model-agnostic deterministic RANSAC engine;
- ``certificate`` — the structured geometric certificate.

Certification policy (see ``veritas.config``): the affine model is the ONLY
accepted certification model. No automatic model selection is performed (the
reference's "auto" ranking is intentionally not migrated) and there is no
homography fallback (the reference's projective solver is reference-only
material). ``verify_affine`` accepts no model argument by design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .certificate import (
    AffineCertificate,
    AffineVerificationResult,  # noqa: F401  (re-exported alias)
    ErrorStatistics,
    VerificationDiagnostics,
)
from .ransac import RansacConfig, ransac

#: Minimal sample size for the affine model (6 DoF require 3 point pairs).
MIN_AFFINE_SAMPLES: int = 3


@dataclass
class AffineVerificationConfig(RansacConfig):
    """Affine verification configuration (RANSAC knobs, affine semantics)."""


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
    """Vectorized 2D Affine Transformation Solver (6 DoF).

    Solves ``[x' y'] = [A t] [x y 1]`` in Hartley-normalised coordinates and
    denormalises back to pixel space.
    """
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
    """Deterministic affine RANSAC: generic engine + affine model pieces."""
    return ransac(
        source_points,
        reference_points,
        min_samples=MIN_AFFINE_SAMPLES,
        fit_model=_fit_affine,
        classify=lambda src, ref, model: classify_matches(src, ref, model, config.reprojection_threshold),
        is_degenerate=_are_points_collinear,
        config=config,
    )


def _certificate(
    transformation: Optional[np.ndarray],
    inlier_mask: np.ndarray,
    error_stats: ErrorStatistics,
    iterations: int,
    converged: bool,
    candidate_count: int,
    config: AffineVerificationConfig,
) -> AffineCertificate:
    """Build the structured affine certificate (see ``veritas.geometry.certificate``)."""
    inlier_count = int(np.sum(inlier_mask)) if len(inlier_mask) else 0
    inlier_ratio = (inlier_count / candidate_count) if candidate_count else 0.0
    diagnostics = VerificationDiagnostics(
        model_name="affine",
        total_matches=candidate_count,
        inlier_count=inlier_count,
        outlier_count=candidate_count - inlier_count,
        inlier_ratio=inlier_ratio,
        iterations_run=iterations,
        converged=bool(converged),
        is_valid=bool(converged and inlier_count >= config.min_inliers),
        error_stats=error_stats,
    )
    return AffineCertificate(
        is_valid=diagnostics.is_valid,
        transformation=transformation,
        inlier_mask=inlier_mask,
        diagnostics=diagnostics,
        source_inliers=np.empty((0, 2), dtype=np.float64),
        reference_inliers=np.empty((0, 2), dtype=np.float64),
        configuration=asdict(config),
    )


def verify_affine(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    config: Optional[Any] = None,
) -> AffineCertificate:
    """Estimate and certify an affine transform between two point sets.

    Affine-only by policy: there is no model argument and no fallback — the
    certification path always invokes affine estimation explicitly.
    """
    cfg = _config(config)
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(reference_points, dtype=np.float64)

    if len(src) != len(dst):
        raise ValueError(f"Point counts mismatch: {len(src)} vs {len(dst)}")

    if len(src) == 0:
        return _certificate(None, np.zeros(0, dtype=bool), ErrorStatistics(), 0, False, 0, cfg)

    if len(src) < MIN_AFFINE_SAMPLES:
        return _certificate(
            None, np.zeros(len(src), dtype=bool), ErrorStatistics(), 0, False, len(src), cfg
        )

    model, mask, iterations, converged = _ransac_affine(src, dst, cfg)
    if model is None:
        return _certificate(
            None, np.zeros(len(src), dtype=bool), ErrorStatistics(), iterations, False, len(src), cfg
        )

    # Describe verified correspondences only; rejected outliers are reported
    # separately in the certificate counts.
    errors = compute_reprojection_errors(src[mask], dst[mask], model)
    error_stats = compute_error_statistics(errors)
    certificate = _certificate(model, mask, error_stats, iterations, converged, len(src), cfg)
    certificate.inlier_indices = np.flatnonzero(mask).astype(np.int64)
    certificate.source_inliers = src[mask]
    certificate.reference_inliers = dst[mask]
    return certificate


def verify_affine_from_correspondences(
    match_result: Any,
    config: Optional[Any] = None,
) -> AffineCertificate:
    """Run affine verification from a matching ``MatchResult``."""
    source = np.asarray(match_result.source_points, dtype=np.float64).reshape(-1, 2)
    reference = np.asarray(match_result.reference_points, dtype=np.float64).reshape(-1, 2)
    return verify_affine(source, reference, config)


def verify_matches(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    config: Optional[Any] = None,
) -> AffineCertificate:
    """Affine-only alias kept for callers of the reference API shape.

    Unlike the reference implementation, this performs NO automatic model
    selection: the certification model is always affine.
    """
    return verify_affine(source_points, reference_points, config)
