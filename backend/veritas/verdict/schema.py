"""Schema for the VERITAS verdict result.

This module defines the `Verdict` enumeration and the `VerdictResult`
dataclass, which encapsulates the outcome of the VERITAS verdict
classification process.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class Verdict(str, Enum):
    """Primary verdict vocabulary for VERITAS."""

    STRONG = "STRONG"
    PARTIAL = "PARTIAL"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass(frozen=True)
class VerdictResult:
    """Comprehensive result of the VERITAS verdict classification.

    This object contains the final verdict, machine-readable rationale codes,
    a summary of supporting and counter evidence, a trace of the decision
    process, heuristic confidence scores, and any identified limitations.
    """

    verdict: Verdict
    rationale_codes: List[str] = field(default_factory=list)
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    counter_evidence: Dict[str, Any] = field(default_factory=dict)
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)
    confidence_status: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the VerdictResult to a dictionary."""
        return {
            "verdict": self.verdict.value,
            "rationale_codes": self.rationale_codes,
            "supporting_evidence": self.supporting_evidence,
            "counter_evidence": self.counter_evidence,
            "decision_trace": self.decision_trace,
            "confidence_status": self.confidence_status,
            "limitations": self.limitations,
        }