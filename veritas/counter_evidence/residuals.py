"""Residual measurements retained as negative geometric evidence.

Computes descriptive reprojection-error statistics for inlier matches. These are
image-space residuals, not probabilities or calibrated confidences.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

#: Sentinel value used when no finite errors are available.
_NO_VALUE: Optional[float] = None


def summarize_residuals(
    errors: Sequence[Union[float, np.floating]],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize reprojection-error statistics for a set of inlier errors.

    Args:
        errors: Reprojection error values (pixel units).
        threshold: Optional inlier acceptance threshold in pixel units.

    Returns:
        A dictionary with count, mean, median, p95, maximum and their stable
        aliases (``mean_reprojection_error``, ``max_reprojection_error``), plus
        ``threshold_exceedance_count`` when a threshold is supplied.
    """
    values = np.asarray(errors, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else _NO_VALUE
    maximum = float(finite.max()) if len(finite) else _NO_VALUE
    return {
        "count": int(len(finite)),
        "mean": mean,
        "median": float(np.median(finite)) if len(finite) else _NO_VALUE,
        "p95": float(np.percentile(finite, 95)) if len(finite) else _NO_VALUE,
        "maximum": maximum,
        # Stable aliases preserve the original verdict/gate contract.
        "mean_reprojection_error": mean,
        "max_reprojection_error": maximum,
        "threshold_exceedance_count": int((finite > threshold).sum()) if threshold is not None else _NO_VALUE,
    }
