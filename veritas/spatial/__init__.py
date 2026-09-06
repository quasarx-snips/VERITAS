"""Spatial evidence over verified correspondences.

P0.5: migrated — ``calculate_spatial_coverage`` (grid occupancy over verified
points). Spatial entropy and density measures arrive in P1. Vocabulary is
descriptive, not probabilistic.
"""

from .coverage import CoverageConfig, calculate_spatial_coverage
from .entropy import calculate_spatial_entropy

__all__ = ["CoverageConfig", "calculate_spatial_coverage", "calculate_spatial_entropy"]
