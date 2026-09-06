"""Evidence-contract serialization tests."""

from veritas.schemas import (
    CounterEvidence, DetectorEvidence, GeometryEvidence, MatchEvidence,
    QuorumEvidence, SpatialEvidence, VerificationEvidence,
)


def test_verification_evidence_serializes_only_computed_fields():
    evidence = VerificationEvidence(
        features={"sift": DetectorEvidence("sift", 4, 4, "float32")},
        matches={"sift": MatchEvidence("sift", 4, 3, {"ratio": 0.8})},
        geometry=GeometryEvidence("affine", True, 3, 3, 1.0, {"rmse": 0.1}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        spatial=SpatialEvidence(4, 4, 3, 3 / 16, 0.4),
        quorum=QuorumEvidence({"sift": "supporting"}, ["sift"], ["sift"], [], 1.0),
        counter_evidence=CounterEvidence({}, {}, {}),
    )
    payload = evidence.to_dict()
    assert payload["geometry"]["model"] == "affine"
    assert payload["features"]["sift"]["descriptor_count"] == 4
    assert payload["evidence_score"] is None
