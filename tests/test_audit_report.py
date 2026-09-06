"""Tests for serializable, complete audit reports."""

import json
import numpy as np

from veritas.audit import AuditReport
from veritas.pipeline import verify_evidence_pair


def test_pipeline_produces_complete_json_round_trip_audit_report():
    image = np.zeros((160, 160), dtype=np.uint8)
    image[20:140, 20:24] = 255
    image[30:34, 30:130] = 255
    result = verify_evidence_pair(image, image, run_id="audit-test-run")
    report = result.audit_report
    encoded = report.to_json()
    restored = AuditReport.from_json(encoded)
    assert json.loads(encoded)["run_id"] == "audit-test-run"
    assert restored.to_dict() == report.to_dict()
    assert set(report.decision_trace) >= {
        "input", "preprocessing", "detector_evidence", "matching", "geometry",
        "spatial_evidence", "counter_evidence", "partial_correspondence",
        "fused_evidence", "verdict", "gate",
    }
    assert report.schema_version
    assert report.geometry["affine_matrix"] is None or isinstance(report.geometry["affine_matrix"], list)
