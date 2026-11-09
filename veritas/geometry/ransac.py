"""VERITAS geometry — generic deterministic RANSAC engine (P0.4 migration).

Extracted from the reference repository's robust-estimation loop so the
certification path composes: this engine knows nothing about the affine model —
it is parameterised by a model fitter, a classification function and an
optional degeneracy guard. Preserved behavior:

- deterministic sampling via ``numpy.random.RandomState(seed)``;
- adaptive iteration bound (standard RANSAC confidence formula);
- degeneracy (e.g. collinearity) rejection of minimal samples;
- optional least-squares refit on the inlier set, followed by re-classification;
- convergence = best inlier count >= ``min_inliers``.

The engine returns a model, an inlier mask, the iteration count and a
convergence flag. It computes no scores and no probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class RansacConfig:
    """Configuration for the deterministic RANSAC engine.

    ``reprojection_threshold`` is expressed in pixel units (image-space
    residual distance). ``random_seed`` makes every run deterministic.
    """

    reprojection_threshold: float = 3.0      # max residual distance for inliers
    confidence: float = 0.99                 # desired RANSAC success level
    max_iterations: int = 2000               # hard iteration cap
    min_inliers: int = 4                     # minimum inliers for convergence
    refit_inliers: bool = True               # least-squares refit on inliers
    random_seed: Optional[int] = 42          # seed for deterministic execution


def ransac(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    min_samples: int,
    fit_model: Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]],
    classify: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    is_degenerate: Optional[Callable[[np.ndarray], bool]] = None,
    config: Optional[RansacConfig] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, int, bool]:
    """Run deterministic RANSAC over point correspondences.

    Returns ``(model, inlier_mask, iterations_run, converged)``.
    """
    cfg = config or RansacConfig()
    num_pts = len(source_points)
    if num_pts < min_samples:
        return None, np.zeros(num_pts, dtype=bool), 0, False

    rng = np.random.RandomState(cfg.random_seed)

    best_inlier_mask = np.zeros(num_pts, dtype=bool)
    best_inlier_count = 0
    best_model = None
    dynamic_max_iters = cfg.max_iterations
    iterations_run = 0

    for _ in range(cfg.max_iterations):
        iterations_run += 1
        if iterations_run > dynamic_max_iters:
            break

        sample_indices = rng.choice(num_pts, size=min_samples, replace=False)
        src_sample = source_points[sample_indices]
        dst_sample = reference_points[sample_indices]

        if is_degenerate is not None and is_degenerate(src_sample):
            continue

        model_candidate = fit_model(src_sample, dst_sample)
        if model_candidate is None or not np.all(np.isfinite(model_candidate)):
            continue

        inlier_mask = classify(source_points, reference_points, model_candidate)
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_model = model_candidate

            w = inlier_count / float(num_pts)
            w_sample = max(w ** min_samples, 1e-12)
            if 1.0 - w_sample > 0.0:
                calc_iters = math.log(1.0 - cfg.confidence) / math.log(1.0 - w_sample)
                dynamic_max_iters = min(cfg.max_iterations, int(math.ceil(calc_iters)))

    if cfg.refit_inliers and best_inlier_count >= min_samples:
        refit_model = fit_model(source_points[best_inlier_mask], reference_points[best_inlier_mask])
        if refit_model is not None and np.all(np.isfinite(refit_model)):
            best_model = refit_model
            best_inlier_mask = classify(source_points, reference_points, best_model)
            best_inlier_count = int(np.sum(best_inlier_mask))

    converged = best_inlier_count >= cfg.min_inliers
    return best_model, best_inlier_mask, iterations_run, converged
