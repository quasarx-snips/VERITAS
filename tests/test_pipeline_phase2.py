import cv2
import numpy as np
from veritas.pipeline import verify_evidence_pair

def test_phase2_pipeline_preserves_three_detector_evidence():
    image = np.zeros((260,260), np.uint8)
    cv2.circle(image, (70,70), 30, 255, 3); cv2.rectangle(image, (140,120), (220,210), 200, 3); cv2.line(image, (15,230), (230,20), 180, 3)
    transformed = cv2.warpAffine(image, np.float32([[1,0,8],[0,1,-5]]), (260,260))
    evidence = verify_evidence_pair(image, transformed, matching_config={"ratio": 0.8})
    assert set(evidence.evidence.features) == {"sift", "orb", "akaze"}
    assert evidence.evidence.geometry.model == "affine"
    assert evidence.evidence.spatial.normalized_entropy is not None
    assert "concentrated" in evidence.evidence.counter_evidence.spatial_concentration