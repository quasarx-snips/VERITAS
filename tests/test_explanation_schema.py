"""Tests for zero-pixel future LLM/TTS explanation contracts."""

import cv2
import numpy as np

from veritas.pipeline import verify_evidence_pair


def test_explanation_payload_is_complete_and_cannot_alter_decision():
    image = np.zeros((180, 180), dtype=np.uint8)
    cv2.line(image, (10, 10), (170, 150), 255, 3)
    cv2.circle(image, (90, 90), 30, 180, 2)
    result = verify_evidence_pair(image, image, run_id="explanation-test")
    payload = result.explanation_payload
    assert payload.verdict == result.verdict.verdict.value
    assert payload.gate_action == result.gate.action.value
    assert payload.headline and payload.key_reasons and payload.recommended_next_action
    assert payload.decision_trace == result.audit.decision_trace
    assert "Do not override the deterministic gate." in payload.prohibited_claims
    assert "pixels" not in payload.to_dict()
