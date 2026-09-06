"""VERITAS matching — descriptor correspondences (P0.3 migration).

Migrated from the reference repository's descriptor matching module. The
matcher deliberately stops at descriptor correspondences: estimating an affine
certificate belongs to ``veritas.geometry``.

Preserved behavior:
- BF and FLANN KNN matching (with LSH for binary/uint8 descriptors);
- Lowe's nearest-neighbour distance ratio test;
- mutual-consistency (reciprocal nearest-neighbour) filtering;
- typed matching per evidence family;
- one-to-one (unique source and reference) selection;
- index-preserving correspondence extraction;
- side-by-side match visualization.

Domain-specific defaults from the reference implementation (e.g. a "crater"
singleton feature class) are removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "BF",              # "BF" or "FLANN"
    "ratio": 0.75,
    "knn_k": 2,
    "mutual_consistency": False,
    "match_same_feature_type": True,
    "unique_train_matches": True,
    "allow_singleton_feature_types": (),
    "norm": None,                # inferred: L2 for float/SIFT, Hamming for uint8
    "flann_trees": 5,
    "flann_checks": 50,
}


@dataclass
class MatchResult:
    """Descriptor correspondences, with indices retained in every record."""

    candidate_matches: List[cv2.DMatch] = field(default_factory=list)
    accepted_matches: List[cv2.DMatch] = field(default_factory=list)
    source_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    reference_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    distances: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))
    number_raw_matches: int = 0
    number_filtered_matches: int = 0
    match_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def raw_matches(self) -> int:
        return self.number_raw_matches

    @property
    def filtered_matches(self) -> int:
        return self.number_filtered_matches

    @property
    def matches(self) -> List[cv2.DMatch]:
        """The descriptor matches that passed configured filters."""
        return self.accepted_matches


def _as_descriptors(descriptors: Any) -> np.ndarray:
    """Normalize descriptors while retaining OpenCV-supported dtype."""
    if descriptors is None:
        return np.empty((0, 0), dtype=np.float32)
    array = np.asarray(descriptors)
    if array.size == 0:
        width = array.shape[1] if array.ndim == 2 else 0
        return np.empty((0, width), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Descriptors must be a two-dimensional array.")
    if array.dtype not in (np.float32, np.uint8):
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _norm_for(descriptors: np.ndarray, configured: Any = None) -> int:
    if configured is not None:
        if isinstance(configured, str):
            name = configured.upper()
            if name in ("L2", "NORM_L2"):
                return cv2.NORM_L2
            if name in ("HAMMING", "NORM_HAMMING"):
                return cv2.NORM_HAMMING
            raise ValueError("norm must be L2 or Hamming")
        return int(configured)
    return cv2.NORM_HAMMING if descriptors.dtype == np.uint8 else cv2.NORM_L2


def match_descriptors(
    source_descriptors: Any,
    reference_descriptors: Any,
    method: str = "BF",
    config: Optional[Mapping[str, Any]] = None,
) -> List[List[cv2.DMatch]]:
    """Return KNN candidate matches. Empty inputs produce an empty list."""
    options = {**DEFAULT_CONFIG, **(config or {})}
    source, reference = _as_descriptors(source_descriptors), _as_descriptors(reference_descriptors)
    if not len(source) or not len(reference):
        return []
    if source.shape[1] != reference.shape[1]:
        raise ValueError("Source and reference descriptor dimensions must match.")

    backend = method.upper()
    k = max(1, min(int(options["knn_k"]), len(reference)))
    if backend == "BF":
        matcher = cv2.BFMatcher(_norm_for(source, options.get("norm")), crossCheck=False)
    elif backend == "FLANN":
        # FLANN's KD-tree works on float descriptors; LSH supports binary ones.
        if source.dtype == np.uint8:
            index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        else:
            source, reference = source.astype(np.float32), reference.astype(np.float32)
            index_params = dict(algorithm=1, trees=int(options["flann_trees"]))
        matcher = cv2.FlannBasedMatcher(index_params, dict(checks=int(options["flann_checks"])))
    else:
        raise ValueError("method must be 'BF' or 'FLANN'.")
    return matcher.knnMatch(source, reference, k=k)


def ratio_test(matches: Iterable[Sequence[cv2.DMatch]], ratio: float = 0.75) -> List[cv2.DMatch]:
    """Apply Lowe's nearest-neighbour distance ratio test."""
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
    """Keep matches whose nearest neighbour is reciprocal in descriptor space."""
    source, reference = _as_descriptors(source_descriptors), _as_descriptors(reference_descriptors)
    flat = _flatten_matches(matches)
    if not flat or not len(source) or not len(reference):
        return []
    reverse = match_descriptors(reference, source, method="BF", config={"knn_k": 1})
    reverse_pairs = {(group[0].queryIdx, group[0].trainIdx) for group in reverse if group}
    return [m for m in flat if (m.trainIdx, m.queryIdx) in reverse_pairs]


def _flatten_matches(matches: Iterable[Any]) -> List[cv2.DMatch]:
    flattened: List[cv2.DMatch] = []
    for item in matches:
        if isinstance(item, cv2.DMatch):
            flattened.append(item)
        elif item:  # permit direct use of KNN output
            flattened.append(item[0])
    return flattened


def _point(keypoint: Any) -> Tuple[float, float]:
    if hasattr(keypoint, "pt"):
        return float(keypoint.pt[0]), float(keypoint.pt[1])
    if hasattr(keypoint, "x") and hasattr(keypoint, "y"):
        return float(keypoint.x), float(keypoint.y)
    if isinstance(keypoint, Mapping):
        return float(keypoint["x"]), float(keypoint["y"])
    return float(keypoint[0]), float(keypoint[1])


def matches_to_correspondences(
    source_keypoints: Sequence[Any],
    reference_keypoints: Sequence[Any],
    matches: Iterable[Any],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Convert matches into point arrays and index-preserving plain records."""
    valid: List[cv2.DMatch] = []
    for m in _flatten_matches(matches):
        if 0 <= m.queryIdx < len(source_keypoints) and 0 <= m.trainIdx < len(reference_keypoints):
            valid.append(m)
    source_points = np.asarray([_point(source_keypoints[m.queryIdx]) for m in valid], dtype=np.float32).reshape(-1, 2)
    reference_points = np.asarray([_point(reference_keypoints[m.trainIdx]) for m in valid], dtype=np.float32).reshape(-1, 2)
    records = [{"source_index": int(m.queryIdx), "reference_index": int(m.trainIdx), "distance": float(m.distance)} for m in valid]
    return source_points, reference_points, records


def _feature_parts(feature_set: Any) -> Tuple[Sequence[Any], Any]:
    """Read a ``(records, descriptors)`` tuple or a feature-set object/dict.

    Only records carrying a ``descriptor_index`` are returned: every descriptor
    row must map to exactly one indexed record.
    """
    if isinstance(feature_set, tuple) and len(feature_set) == 2:
        keypoints, descriptors = feature_set
    elif isinstance(feature_set, Mapping):
        descriptors = feature_set.get("descriptors")
        keypoints = feature_set.get("keypoints", feature_set.get("features", []))
    else:
        descriptors = getattr(feature_set, "descriptors", None)
        keypoints = getattr(feature_set, "keypoints", getattr(feature_set, "features", []))
    if descriptors is None:
        raise ValueError("FeatureSet must provide a 'descriptors' array.")
    indexed = [p for p in keypoints if getattr(p, "descriptor_index", None) is not None]
    if indexed:
        indexed.sort(key=lambda p: p.descriptor_index)
        keypoints = indexed
    return keypoints, descriptors


def _feature_type(keypoint: Any) -> str:
    """Return a matching class, with untyped keypoints sharing one class."""
    family = getattr(keypoint, "evidence_family", None)
    if family is None and isinstance(keypoint, Mapping):
        family = keypoint.get("evidence_family", keypoint.get("feature_type"))
    if family is not None:
        return str(family)
    return "__untyped__"


def _typed_candidate_matches(
    source_descriptors: Any,
    reference_descriptors: Any,
    source_keypoints: Sequence[Any],
    reference_keypoints: Sequence[Any],
    options: Mapping[str, Any],
) -> List[List[cv2.DMatch]]:
    """Run KNN matching per evidence family when labels are available."""
    source = _as_descriptors(source_descriptors)
    reference = _as_descriptors(reference_descriptors)
    if not options["match_same_feature_type"]:
        return match_descriptors(source, reference, options["method"], options)
    source_groups: Dict[str, List[int]] = {}
    reference_groups: Dict[str, List[int]] = {}
    for index, keypoint in enumerate(source_keypoints):
        source_groups.setdefault(_feature_type(keypoint), []).append(index)
    for index, keypoint in enumerate(reference_keypoints):
        reference_groups.setdefault(_feature_type(keypoint), []).append(index)
    candidates: List[List[cv2.DMatch]] = []
    for feature_class, source_indices in source_groups.items():
        reference_indices = reference_groups.get(feature_class, [])
        if not reference_indices:
            continue
        local = match_descriptors(source[source_indices], reference[reference_indices], options["method"], options)
        for neighbours in local:
            candidates.append(
                [
                    cv2.DMatch(source_indices[match.queryIdx], reference_indices[match.trainIdx], 0, match.distance)
                    for match in neighbours
                ]
            )
    return candidates


def _unique_train_matches(matches: Iterable[cv2.DMatch]) -> List[cv2.DMatch]:
    """Keep the lowest-distance match for each source and reference index."""
    selected: List[cv2.DMatch] = []
    used_source, used_reference = set(), set()
    for match in sorted(matches, key=lambda item: item.distance):
        if match.queryIdx not in used_source and match.trainIdx not in used_reference:
            selected.append(match)
            used_source.add(match.queryIdx)
            used_reference.add(match.trainIdx)
    return selected


def match_feature_sets(
    source_features: Any,
    reference_features: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> MatchResult:
    """Match two feature sets and report descriptor-only correspondences."""
    options = {**DEFAULT_CONFIG, **(config or {})}
    source_keypoints, source_descriptors = _feature_parts(source_features)
    reference_keypoints, reference_descriptors = _feature_parts(reference_features)
    raw = _typed_candidate_matches(
        source_descriptors, reference_descriptors,
        source_keypoints, reference_keypoints, options,
    )
    filtered = ratio_test(raw, float(options["ratio"])) if int(options["knn_k"]) >= 2 else _flatten_matches(raw)
    # A rare evidence family can have exactly one reference candidate. Lowe's
    # test cannot score that case; retain it only for explicitly configured
    # classes and let one-to-one selection decide.
    singleton_types = set(options.get("allow_singleton_feature_types", ()))
    if singleton_types and int(options["knn_k"]) >= 2:
        filtered.extend(
            neighbours[0] for neighbours in raw
            if len(neighbours) == 1 and _feature_type(source_keypoints[neighbours[0].queryIdx]) in singleton_types
        )
    if options["mutual_consistency"]:
        filtered = mutual_consistency_filter(source_descriptors, reference_descriptors, filtered)
    if options["unique_train_matches"]:
        filtered = _unique_train_matches(filtered)
    source_points, reference_points, records = matches_to_correspondences(source_keypoints, reference_keypoints, filtered)
    distances = np.asarray([r["distance"] for r in records], dtype=np.float32)
    result = MatchResult(
        candidate_matches=_flatten_matches(raw),
        accepted_matches=filtered,
        source_points=source_points,
        reference_points=reference_points,
        distances=distances,
        number_raw_matches=len(raw),
        number_filtered_matches=len(filtered),
        match_records=records,
    )
    retained = 100.0 * result.number_filtered_matches / result.number_raw_matches if result.number_raw_matches else 0.0
    print(f"raw matches: {result.number_raw_matches}")
    print(f"filtered matches: {result.number_filtered_matches}")
    print(f"percentage retained: {retained:.1f}%")
    return result


def _display_image(image: Any) -> np.ndarray:
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array.copy()


def visualize_matches(
    source: Any,
    reference: Any,
    match_result: MatchResult,
    max_matches: int = 200,
) -> np.ndarray:
    """Return a match image; it neither opens a GUI window nor writes a file."""
    src, ref = _display_image(source), _display_image(reference)
    height, width = max(src.shape[0], ref.shape[0]), src.shape[1] + ref.shape[1]
    canvas = np.zeros((height, width, 3), dtype=src.dtype)
    canvas[: src.shape[0], : src.shape[1]] = src
    canvas[: ref.shape[0], src.shape[1]:] = ref
    for i, (sp, rp) in enumerate(zip(match_result.source_points[:max_matches], match_result.reference_points[:max_matches])):
        color = tuple(int(v) for v in np.random.default_rng(i).integers(80, 256, 3))
        a, b = tuple(np.round(sp).astype(int)), tuple(np.round(rp + [src.shape[1], 0]).astype(int))
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, b, 3, color, 1, cv2.LINE_AA)
    return canvas