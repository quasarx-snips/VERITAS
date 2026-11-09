"""VERITAS matching â€” descriptor correspondences (P0.3 migration).

Migrated from the reference repository's descriptor matching module. This is
the raw-matching and orchestration half; the decision filters live in
``filtering``. The pipeline is:

    DescriptorMatcher (BF / FLANN)
        â†“ raw candidate matches
    ratio filter â†’ mutual consistency filter â†’ unique train match filter
        â†“
    accepted correspondence set (``MatchResult``)

Distance rule (respected, never silently broken):
- floating-point descriptors (e.g. SIFT)  â†’ L2 (+ FLANN KD-tree);
- binary uint8 descriptors (ORB/AKAZE)    â†’ Hamming (+ FLANN LSH).

Preserved behavior: BF and FLANN KNN matching, Lowe's ratio test, reciprocal
(mutual-consistency) filtering, typed matching per evidence family, one-to-one
(unique source and reference) selection, index-preserving correspondence
extraction, side-by-side match visualization. Domain-specific defaults from the
reference implementation (e.g. its domain-specific singleton feature class) are
removed. This module must not know about verdicts, entropy, gates, or LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .filtering import flatten_matches, mutual_consistency_filter, ratio_test, unique_train_matches

DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "BF",              # "BF" or "FLANN"
    "ratio": 0.75,
    "knn_k": 2,
    "mutual_consistency": False,
    "match_same_feature_type": True,
    "unique_train_matches": True,
    "allow_singleton_feature_types": (),
    "norm": None,                # inferred: L2 for float, Hamming for uint8
    "flann_trees": 5,
    "flann_checks": 50,
}


@dataclass
class MatchResult:
    """Descriptor correspondences, with indices retained in every record.

    Carries everything later geometry needs â€” point arrays, accepted index
    pairs, raw candidates, distances, and filtering diagnostics â€” and no
    visualization state (``visualize_matches`` renders on demand).
    """

    candidate_count: int = 0
    accepted_count: int = 0
    source_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    reference_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    distances: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))
    source_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), np.int64))
    reference_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), np.int64))
    candidate_matches: List[cv2.DMatch] = field(default_factory=list)
    accepted_matches: List[cv2.DMatch] = field(default_factory=list)
    match_records: List[Dict[str, Any]] = field(default_factory=list)
    descriptor_family: Optional[str] = None
    filter_diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def number_raw_matches(self) -> int:
        """Backwards-compatible alias for :attr:`candidate_count`."""
        return self.candidate_count

    @property
    def number_filtered_matches(self) -> int:
        """Backwards-compatible alias for :attr:`accepted_count`."""
        return self.accepted_count

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
    """Pick the distance norm that matches the descriptor type."""
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
    for m in flatten_matches(matches):
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
    flat_raw = flatten_matches(raw)
    use_ratio = int(options["knn_k"]) >= 2
    filtered = ratio_test(raw, float(options["ratio"])) if use_ratio else list(flat_raw)
    after_ratio = len(filtered)
    # A rare evidence family can have exactly one reference candidate. Lowe's
    # test cannot score that case; retain it only for explicitly configured
    # classes and let one-to-one selection decide.
    singleton_types = set(options.get("allow_singleton_feature_types", ()))
    retained_singletons = 0
    if singleton_types and use_ratio:
        singletons = [
            neighbours[0] for neighbours in raw
            if len(neighbours) == 1 and _feature_type(source_keypoints[neighbours[0].queryIdx]) in singleton_types
        ]
        filtered.extend(singletons)
        retained_singletons = len(singletons)
    if options["mutual_consistency"]:
        filtered = mutual_consistency_filter(source_descriptors, reference_descriptors, filtered)
    after_mutual = len(filtered)
    if options["unique_train_matches"]:
        filtered = unique_train_matches(filtered)
    after_unique = len(filtered)

    source_points, reference_points, records = matches_to_correspondences(source_keypoints, reference_keypoints, filtered)
    distances = np.asarray([r["distance"] for r in records], dtype=np.float32)
    families = sorted({_feature_type(source_keypoints[r["source_index"]]) for r in records})
    descriptor_family = families[0] if len(families) == 1 and families[0] != "__untyped__" else None
    diagnostics: Dict[str, Any] = {
        "configured": {
            name: options[name]
            for name in ("method", "ratio", "knn_k", "mutual_consistency", "match_same_feature_type", "unique_train_matches")
        },
        "candidate_count": len(flat_raw),
        "after_ratio_test": after_ratio,
        "singleton_retained": retained_singletons,
        "after_mutual_consistency": after_mutual,
        "after_unique_train": after_unique,
        "removed_by_ratio": len(flat_raw) - after_ratio,
        "removed_by_mutual": after_ratio + retained_singletons - after_mutual,
        "removed_by_unique": after_mutual - after_unique,
        "source_descriptor_families": families,
    }
    result = MatchResult(
        candidate_count=len(raw),
        accepted_count=len(filtered),
        source_points=source_points,
        reference_points=reference_points,
        distances=distances,
        source_indices=np.asarray([r["source_index"] for r in records], dtype=np.int64),
        reference_indices=np.asarray([r["reference_index"] for r in records], dtype=np.int64),
        candidate_matches=flat_raw,
        accepted_matches=filtered,
        match_records=records,
        descriptor_family=descriptor_family,
        filter_diagnostics=diagnostics,
    )
    retained = 100.0 * result.accepted_count / result.candidate_count if result.candidate_count else 0.0
    print(f"raw matches: {result.candidate_count}")
    print(f"filtered matches: {result.accepted_count}")
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
