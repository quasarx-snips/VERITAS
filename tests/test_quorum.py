from veritas.features import assess_quorum
from veritas.schemas import GeometryEvidence


def _geometry(certified):
    return GeometryEvidence("affine", certified, 5, 5 if certified else 0, 1.0 if certified else 0.0, {}, None)


def test_quorum_distinguishes_unavailable_from_disagreement():
    result = assess_quorum({"sift": _geometry(True), "orb": _geometry(False), "akaze": None})
    assert result.supporting_detectors == ["sift"]
    assert result.disagreeing_detectors == ["orb"]
    assert result.detector_results["akaze"] == "unavailable"
    assert result.quorum_strength == 0.5
