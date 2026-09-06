"""Phase 3C integration test for the complete backend contract."""

import json

import cv2
import numpy as np

from veritas.pipeline import verify_evidence_pair


def test_complete_pipeline_preserves_evidence_and_decision_trace():
    image = np.zeros((220, 220), dtype=np.uint8)
    cv2.circle(image, (55, 55), 25, 255, 3)
    cv2.rectangle(image, (120, 100), (195, 190), 200, 3)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, -4]]), (220, 220))
    result = verify_evidence_pair(image, shifted, run_id="pipeline-decision-test")
    report = result.to_dict()
    assert set(report) == {"evidence", "partial_correspondence", "verdict", "gate", "audit", "explanation_payload"}
    assert set(report["evidence"]["features"]) == {"sift", "orb", "akaze"}
    assert report["audit"]["decision_trace"]["gate"] == report["gate"]
    assert json.loads(json.dumps(report))["audit"]["run_id"] == "pipeline-decision-test"
