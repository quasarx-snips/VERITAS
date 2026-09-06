"""VERITAS SIFT feature-evidence tests (P0.2 migration)."""

import json

import cv2
import numpy as np

from veritas.features import FeatureStore, SiftDetector
from veritas.schemas import EVIDENCE_FAMILY_SIFT, FeatureEvidence


def _synthetic_image(shape=(240, 240)) -> np.ndarray:
    image = np.zeros(shape, np.uint8)
    cv2.circle(image, (80, 80), 25, 255, 3)
    cv2.rectangle(image, (130, 120), (190, 180), 180, 3)
    cv2.line(image, (30, 200), (205, 35), 220, 3)
    cv2.circle(image, (150, 60), 40, 200, 3)
    return image


def test_detect_returns_evidence_records_and_aligned_descriptors():
    detector = SiftDetector(n_features=2000)
    features, descriptors = detector.detect(_synthetic_image())
    assert len(features) > 0
    assert descriptors.shape == (len(features), 128)
    assert descriptors.dtype == np.float32
    for index, feature in enumerate(features):
        assert feature.evidence_family == EVIDENCE_FAMILY_SIFT
        assert feature.descriptor_index == index
        assert feature.x > 0.0 and feature.y > 0.0
        assert isinstance(feature, FeatureEvidence)


def test_plain_sift_is_identity_when_root_disabled():
    detector = SiftDetector()
    _, descriptors = detector.detect(_synthetic_image())
    normalized = detector._normalize_descriptors(descriptors)
    assert np.array_equal(normalized, descriptors)


def test_root_sift_alters_descriptor_vectors():
    plain = SiftDetector(root_sift=False)
    rooted = SiftDetector(root_sift=True)
    image = _synthetic_image()
    _, plain_desc = plain.detect(image)
    _, root_desc = rooted.detect(image)
    assert plain_desc.shape == root_desc.shape
    assert not np.allclose(plain_desc, root_desc)
    # RootSIFT output stays in the same numeric family and is finite.
    assert np.isfinite(root_desc).all()


def test_describe_features_assigns_descriptor_indices():
    detector = SiftDetector()
    image = _synthetic_image()
    features, _ = detector.detect(image)
    sample = list(features[:5])
    descriptors = detector.describe_features(image, sample)
    assert descriptors.ndim == 2
    assert descriptors.shape[1] == 128
    assigned = [f for f in sample if f.descriptor_index is not None]
    assert len(assigned) > 0


def test_describe_features_returns_empty_for_no_features():
    detector = SiftDetector()
    descriptors = detector.describe_features(_synthetic_image(), [])
    assert descriptors.shape == (0, 128)


def test_descriptor_offset_shifts_indices():
    detector = SiftDetector()
    features, _ = detector.detect(_synthetic_image(), descriptor_offset=100)
    assert all(f.descriptor_index >= 100 for f in features)


def test_feature_store_roundtrip(tmp_path):
    detector = SiftDetector()
    image = _synthetic_image()
    features, descriptors = detector.detect(image)
    features_path = tmp_path / "features.json"
    descriptors_path = tmp_path / "descriptors.npy"
    FeatureStore.save_features(
        features_path,
        features,
        image_shape=image.shape,
        metadata={"descriptor_dim": 128},
    )
    FeatureStore.save_descriptors(descriptors_path, descriptors)
    assert features_path.is_file()
    assert descriptors_path.is_file()
    payload = json.loads(features_path.read_text(encoding="utf-8"))
    assert payload["feature_count"] == len(features)
    assert payload["image_shape"] == [240, 240]
    assert np.array_equal(np.load(str(descriptors_path)), descriptors)