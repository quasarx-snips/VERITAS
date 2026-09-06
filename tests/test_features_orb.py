import cv2
import numpy as np

from veritas.features import OrbDetector
from veritas.matching import match_feature_sets


def _image():
    image = np.zeros((240, 240), np.uint8)
    cv2.circle(image, (70, 80), 30, 255, 3)
    cv2.rectangle(image, (130, 100), (200, 190), 200, 3)
    cv2.line(image, (20, 220), (210, 20), 180, 3)
    return image


def test_orb_extracts_binary_metadata_and_matches():
    detector = OrbDetector(n_features=1000)
    image = _image()
    features, descriptors = detector.detect(image)
    assert len(features) > 0
    assert descriptors.dtype == np.uint8 and descriptors.shape == (len(features), 32)
    assert all(feature.evidence_family == "orb" and feature.descriptor_index == i for i, feature in enumerate(features))
    result = match_feature_sets((features, descriptors), (features, descriptors.copy()), {"ratio": 0.8})
    # Repeated binary patterns can be ambiguous under Lowe filtering; retain
    # actual accepted matches without fabricating a one-to-one total.
    assert result.accepted_count > 0
    assert result.descriptor_family == "orb"


def test_orb_blank_image_has_empty_binary_descriptors():
    features, descriptors = OrbDetector().detect(np.zeros((80, 80), np.uint8))
    assert features == []
    assert descriptors.shape == (0, 32) and descriptors.dtype == np.uint8
