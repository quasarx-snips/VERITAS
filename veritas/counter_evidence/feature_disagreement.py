"""Detector support disagreement facts."""
from veritas.schemas import QuorumEvidence

def summarize_feature_disagreement(quorum: QuorumEvidence):
    return {"available_detectors": list(quorum.available_detectors), "supporting_detectors": list(quorum.supporting_detectors),
            "disagreeing_detectors": list(quorum.disagreeing_detectors), "unavailable_detectors": [k for k,v in quorum.detector_results.items() if v == "unavailable"]}
