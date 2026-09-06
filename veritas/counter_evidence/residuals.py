"""Residual measurements retained as negative geometric evidence."""
import numpy as np

def summarize_residuals(errors, threshold=None):
    values = np.asarray(errors, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else None
    maximum = float(finite.max()) if len(finite) else None
    return {"count": int(len(finite)), "mean": mean,
            "median": float(np.median(finite)) if len(finite) else None,
            "p95": float(np.percentile(finite, 95)) if len(finite) else None,
            "maximum": maximum,
            # Stable aliases preserve the original verdict/gate contract.
            "mean_reprojection_error": mean,
            "max_reprojection_error": maximum,
            "threshold_exceedance_count": int((finite > threshold).sum()) if threshold is not None else None}
