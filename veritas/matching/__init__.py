"""Descriptor matching primitives.

P0.3: migrated — the reference matcher (BF/FLANN KNN with LSH for binary
descriptors, Lowe's ratio test, mutual-consistency filtering, typed matching
per evidence family, one-to-one selection, correspondence extraction,
visualization), split by responsibility:

- ``descriptor_matching`` — raw candidate generation + ``MatchResult``;
- ``filtering`` — ratio / mutual-consistency / unique-train filters.

Matching stops at descriptor correspondences; geometric certification belongs
to ``veritas.geometry``.
"""

from .descriptor_matching import (
    DEFAULT_CONFIG,
    MatchResult,
    match_descriptors,
    match_feature_sets,
    matches_to_correspondences,
    visualize_matches,
)
from .filtering import (
    flatten_matches,
    mutual_consistency_filter,
    ratio_test,
    unique_train_matches,
)

__all__ = [
    "DEFAULT_CONFIG",
    "MatchResult",
    "flatten_matches",
    "match_descriptors",
    "match_feature_sets",
    "matches_to_correspondences",
    "mutual_consistency_filter",
    "ratio_test",
    "unique_train_matches",
    "visualize_matches",
]
