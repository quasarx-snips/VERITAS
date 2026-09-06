"""Descriptor matching primitives.

P0.3: migrated — the reference matcher (MatchResult, BF/FLANN KNN with LSH for
binary descriptors, Lowe's ratio test, mutual-consistency filtering, typed
matching per evidence family, one-to-one selection, correspondence extraction,
visualization). Matching stops at descriptor correspondences; geometric
certification belongs to ``veritas.geometry``.
"""

from .core import (
    DEFAULT_CONFIG,
    MatchResult,
    match_descriptors,
    match_feature_sets,
    matches_to_correspondences,
    mutual_consistency_filter,
    ratio_test,
    visualize_matches,
)

__all__ = [
    "DEFAULT_CONFIG",
    "MatchResult",
    "match_descriptors",
    "match_feature_sets",
    "matches_to_correspondences",
    "mutual_consistency_filter",
    "ratio_test",
    "visualize_matches",
]