"""Detector support disagreement facts."""
from veritas.schemas import QuorumEvidence

def summarize_feature_disagreement(quorum: QuorumEvidence):
    available = list(quorum.available_detectors)
    disagreeing = list(quorum.disagreeing_detectors)
    return {"available_detectors": available, "supporting_detectors": list(quorum.supporting_detectors),
            "disagreeing_detectors": disagreeing, "unavailable_detectors": [k for k,v in quorum.detector_results.items() if v == "unavailable"],
            "disagreement_score": len(disagreeing) / len(available) if available else 0.0}
