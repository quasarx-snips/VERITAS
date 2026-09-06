"""Per-detector geometric support summary; this is not a majority verdict."""

from typing import Mapping, Optional

from ..schemas import GeometryEvidence, QuorumEvidence


def assess_quorum(results: Mapping[str, Optional[GeometryEvidence]]) -> QuorumEvidence:
    """Classify detector evidence as supporting, disagreeing, or unavailable."""
    states = {name: ("unavailable" if item is None else ("supporting" if item.certified else "disagreeing"))
              for name, item in results.items()}
    available = [name for name, state in states.items() if state != "unavailable"]
    supporting = [name for name, state in states.items() if state == "supporting"]
    disagreeing = [name for name, state in states.items() if state == "disagreeing"]
    return QuorumEvidence(states, available, supporting, disagreeing, len(supporting) / len(available) if available else 0.0)
