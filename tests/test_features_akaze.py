import cv2
import numpy as np

from veritas.features import AkazeDetector
from veritas.matching import match_feature_sets


def test_akaze_extracts_binary_metadata_and_matches():
    image = np.zeros((240, 240), np.uint8)
    cv2.circle(image, (70, 80), 30, 255, 3)
    cv2.rectangle(image, (130, 100), (200, 190), 200, 3)
    cv2.line(image, (20, 220), (210, 20), 180, 3)
    features, descriptors = AkazeDetector().detect(image)
    assert len(features) > 0 and descriptors.dtype == np.uint8
    assert descriptors.shape[0] == len(features)
    assert all(item.evidence_family == "akaze" for item in features)
    result = match_feature_sets((features, descriptors), (features, descriptors.copy()), {"ratio": 0.8})
    assert result.accepted_count > 0 and result.descriptor_family == "akaze"


def test_akaze_blank_image_is_safe():
    features, descriptors = AkazeDetector().detect(np.zeros((80, 80), np.uint8))
    assert features == [] and descriptors.dtype == np.uint8 and descriptors.shape[0] == 0
