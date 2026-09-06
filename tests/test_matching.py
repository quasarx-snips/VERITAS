"""VERITAS descriptor matching tests (P0.3 migration)."""

from types import SimpleNamespace

import cv2
import numpy as np

from veritas.matching import (
    match_descriptors,
    match_feature_sets,
    matches_to_correspondences,
    ratio_test,
    visualize_matches,
)


def test_synthetic_matching_retains_indices_and_points():
    rng = np.random.default_rng(4)
    descriptors = rng.normal(size=(12, 128)).astype(np.float32)
    source = SimpleNamespace(keypoints=[(float(i), float(i + 1)) for i in range(12)], descriptors=descriptors)
    reference = SimpleNamespace(keypoints=[(float(i + 10), float(i + 20)) for i in range(12)], descriptors=descriptors.copy())

    result = match_feature_sets(source, reference, {"ratio": 0.8, "mutual_consistency": True})

    assert result.number_raw_matches == 12
    assert result.number_filtered_matches == 12
    assert result.source_points.dtype == np.float32
    assert np.array_equal(result.source_points, np.asarray(source.keypoints, np.float32))
    assert [r["source_index"] for r in result.match_records] == list(range(12))
    assert [r["reference_index"] for r in result.match_records] == list(range(12))


def test_empty_descriptors_and_ratio_with_one_neighbour_are_safe():
    assert match_descriptors(None, np.empty((2, 128), np.float32)) == []
    assert ratio_test([]) == []
    source, reference, records = matches_to_correspondences([], [], [])
    assert source.shape == reference.shape == (0, 2)
    assert records == []


def test_feature_set_uses_descriptor_indexed_records_only():
    descriptors = np.eye(3, dtype=np.float32)
    # SIFT evidence records come first with a descriptor_index; unindexed
    # records (records without descriptors) must not participate.
    source_features = [
        SimpleNamespace(evidence_family="sift", x=-1.0, y=-1.0, descriptor_index=None),
        SimpleNamespace(evidence_family="sift", x=10.0, y=20.0, descriptor_index=0),
        SimpleNamespace(evidence_family="sift", x=30.0, y=40.0, descriptor_index=1),
        SimpleNamespace(evidence_family="sift", x=50.0, y=60.0, descriptor_index=2),
    ]
    reference_features = [
        SimpleNamespace(evidence_family="sift", x=-2.0, y=-2.0, descriptor_index=None),
        SimpleNamespace(evidence_family="sift", x=15.0, y=25.0, descriptor_index=0),
        SimpleNamespace(evidence_family="sift", x=35.0, y=45.0, descriptor_index=1),
        SimpleNamespace(evidence_family="sift", x=55.0, y=65.0, descriptor_index=2),
    ]
    result = match_feature_sets((source_features, descriptors), (reference_features, descriptors), {"ratio": 0.8})

    assert np.array_equal(result.source_points, np.array([[10, 20], [30, 40], [50, 60]], np.float32))
    assert np.array_equal(result.reference_points, np.array([[15, 25], [35, 45], [55, 65]], np.float32))


def test_typed_matching_requires_matching_families():
    """An evidence family absent from one side must produce no cross-family pairs."""
    rng = np.random.default_rng(11)
    descriptors = rng.normal(size=(8, 128)).astype(np.float32)
    source = SimpleNamespace(
        keypoints=[
            SimpleNamespace(evidence_family="sift", x=1.0, y=1.0, descriptor_index=i) for i in range(4)
        ] + [
            SimpleNamespace(evidence_family="orb", x=2.0, y=2.0, descriptor_index=4 + i) for i in range(4)
        ],
        descriptors=descriptors,
    )
    reference = SimpleNamespace(
        keypoints=[SimpleNamespace(evidence_family="sift", x=1.0, y=1.0, descriptor_index=i) for i in range(4)],
        descriptors=descriptors[:4].copy(),
    )
    result = match_feature_sets(source, reference, {"ratio": 0.95})
    assert result.number_filtered_matches == 4
    for record in result.match_records:
        assert source.keypoints[record["source_index"]].evidence_family == "sift"
        assert reference.keypoints[record["reference_index"]].evidence_family == "sift"


def test_hamming_norm_inferred_for_binary_descriptors():
    source = np.eye(4, dtype=np.uint8)
    result = match_descriptors(source, source.copy(), method="BF", config={"knn_k": 1})
    assert len(result) == 4
    assert all(m[0].distance == 0 for m in result)


def test_real_image_sift_matching_and_visualization():
    image = np.zeros((240, 240), np.uint8)
    cv2.circle(image, (80, 80), 25, 255, 3)
    cv2.rectangle(image, (130, 120), (190, 180), 180, 3)
    cv2.line(image, (30, 200), (205, 35), 220, 3)
    transformed = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 5]]), (240, 240))
    sift = cv2.SIFT_create()
    source_kp, source_desc = sift.detectAndCompute(image, None)
    reference_kp, reference_desc = sift.detectAndCompute(transformed, None)

    result = match_feature_sets(
        SimpleNamespace(keypoints=source_kp, descriptors=source_desc),
        SimpleNamespace(keypoints=reference_kp, descriptors=reference_desc),
        {"method": "FLANN", "ratio": 0.8},
    )
    visualization = visualize_matches(image, transformed, result)

    assert result.number_raw_matches > 0
    assert result.number_filtered_matches > 0
    assert visualization.shape == (240, 480, 3)