"""VERITAS matching — match filtering primitives (P0.3 migration).

Responsibility split (raw matching lives in ``descriptor_matching``): this
module owns everything that decides WHICH raw candidate correspondences
survive:

- ``ratio_test`` — Lowe's nearest-neighbour distance ratio test;
- ``mutual_consistency_filter`` — keep only reciprocal nearest neighbours;
- ``unique_train_matches`` — enforce a one-to-one source/reference assignment.

This module knows nothing about geometry, verdicts, entropy, gating, or LLMs;
its only job is correspondence filtering. Domain-specific defaults from the
reference implementation are removed.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import cv2


def flatten_matches(matches: Iterable[Any]) -> List[cv2.DMatch]:
    """Flatten KNN groups (or already-flat matches) into a single list."""
    flattened: List[cv2.DMatch] = []
    for item in matches:
        if isinstance(item, cv2.DMatch):
            flattened.append(item)
        elif item:  # permit direct use of KNN output
            flattened.append(item[0])
    return flattened


def ratio_test(matches: Iterable[Sequence[cv2.DMatch]], ratio: float = 0.75) -> List[cv2.DMatch]:
    """Apply Lowe's nearest-neighbour distance ratio test.

    A candidate is accepted when ``best.distance < ratio * runner_up.distance``
    with ``0 < ratio <= 1``: smaller ratios are stricter. A neighbour list with
    a single entry carries no ambiguity estimate and is therefore NOT silently
    accepted (callers may retain such singletons explicitly).
    """
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1].")
    good: List[cv2.DMatch] = []
    for neighbours in matches:
        if not neighbours:
            continue
        if len(neighbours) == 1:
            # No ambiguity estimate exists, so do not silently treat it as good.
            continue
        best, runner_up = neighbours[0], neighbours[1]
        if best.distance < ratio * runner_up.distance:
            good.append(best)
    return good


def mutual_consistency_filter(
    source_descriptors: Any,
    reference_descriptors: Any,
    matches: Iterable[Any],
) -> List[cv2.DMatch]:
    """Keep matches whose nearest neighbour is reciprocal in descriptor space.

    A reverse nearest-neighbour query (reference -> source) is run on the raw
    descriptors; a match survives only when the reverse direction maps it back
    consistently. The descriptor-matching import is deferred to avoid an import
    cycle (``descriptor_matching`` is the orchestrator importing this module).
    """
    from .descriptor_matching import _as_descriptors, match_descriptors

    source, reference = _as_descriptors(source_descriptors), _as_descriptors(reference_descriptors)
    flat = flatten_matches(matches)
    if not flat or not len(source) or not len(reference):
        return []
    reverse = match_descriptors(reference, source, method="BF", config={"knn_k": 1})
    reverse_pairs = {(group[0].queryIdx, group[0].trainIdx) for group in reverse if group}
    return [m for m in flat if (m.trainIdx, m.queryIdx) in reverse_pairs]


def unique_train_matches(matches: Iterable[cv2.DMatch]) -> List[cv2.DMatch]:
    """Keep the lowest-distance match for each source and reference index."""
    selected: List[cv2.DMatch] = []
    used_source, used_reference = set(), set()
    for match in sorted(matches, key=lambda item: item.distance):
        if match.queryIdx not in used_source and match.trainIdx not in used_reference:
            selected.append(match)
            used_source.add(match.queryIdx)
            used_reference.add(match.trainIdx)
    return selected
