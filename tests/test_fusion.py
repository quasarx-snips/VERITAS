from veritas.verification import fuse_evidence
from veritas.schemas import CounterEvidence, GeometryEvidence, QuorumEvidence, SpatialEvidence

def test_fusion_keeps_raw_evidence_without_probability():
    output = fuse_evidence({}, {}, GeometryEvidence("affine", False, 0, 0, 0, {}, None), SpatialEvidence(4,4,0,0,0), QuorumEvidence({},[],[],[],0), CounterEvidence({}, {}, {}))
    assert output.evidence_score is None and output.geometry.model == "affine"
