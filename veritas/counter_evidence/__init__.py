"""Counter-evidence checks.

P1.10: checks that argue against a verified correspondence (e.g., conflicting
evidence families, degenerate inlier geometry, weak spatial distribution).
Nothing is implemented before its migration/feature commit.
"""
from .residuals import summarize_residuals
from .feature_disagreement import summarize_feature_disagreement
from .spatial_concentration import summarize_spatial_concentration

__all__ = ["summarize_residuals", "summarize_feature_disagreement", "summarize_spatial_concentration"]
