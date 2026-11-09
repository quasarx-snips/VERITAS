"""Detector support disagreement facts.

Summarizes per-detector support states from a :class:`~veritas.schemas.QuorumEvidence`
record. Disagreement is purely descriptive: unavailable evidence is never treated
as disagreement.
"""

from __future__ import annotations

from typing import Any, Dict

from veritas.schemas import QuorumEvidence


def summarize_feature_disagreement(quorum: QuorumEvidence) -> Dict[str, Any]:
    """Return disagreement metrics derived from a quorum evidence record.

    Args:
        quorum: The aggregated per-detector support states.

    Returns:
        A dictionary with ``available_detectors``, ``supporting_detectors``,
        ``disagreeing_detectors``, ``unavailable_detectors`` and a
        ``disagreement_score`` in ``[0.0, 1.0]``.
    """
    available = list(quorum.available_detectors)
    disagreeing = list(quorum.disagreeing_detectors)
    return {
        "available_detectors": available,
        "supporting_detectors": list(quorum.supporting_detectors),
        "disagreeing_detectors": disagreeing,
        "unavailable_detectors": [
            key for key, value in quorum.detector_results.items() if value == "unavailable"
        ],
        "disagreement_score": len(disagreeing) / len(available) if available else 0.0,
    }
