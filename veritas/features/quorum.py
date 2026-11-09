"""Per-detector geometric support summary; this is not a majority verdict."""

from __future__ import annotations

from typing import Mapping, Optional

from ..schemas import GeometryEvidence, QuorumEvidence


def assess_quorum(results: Mapping[str, Optional[GeometryEvidence]]) -> QuorumEvidence:
    """Classify each detector's evidence as supporting, disagreeing, or unavailable.

    Args:
        results: Mapping of detector name to its optional geometric certificate.

    Returns:
        A :class:`~veritas.schemas.QuorumEvidence` with per-detector states and a
        quorum strength in ``[0.0, 1.0]``.
    """
    states = {
        name: (
            "unavailable"
            if item is None
            else ("supporting" if item.certified else "disagreeing")
        )
        for name, item in results.items()
    }
    available = [name for name, state in states.items() if state != "unavailable"]
    supporting = [name for name, state in states.items() if state == "supporting"]
    disagreeing = [name for name, state in states.items() if state == "disagreeing"]
    return QuorumEvidence(
        states,
        available,
        supporting,
        disagreeing,
        len(supporting) / len(available) if available else 0.0,
    )
