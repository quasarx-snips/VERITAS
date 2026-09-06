"""Provider-independent, zero-pixel contract for future explanation adapters."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class ExplanationPayload:
    """All structured context a future LLM or TTS adapter may consume.

    This is data only.  It has no authority to classify evidence or alter a
    safety-gate result.
    """

    schema_version: str
    task: str
    headline: str
    verdict: str
    gate_action: str
    key_reasons: List[str]
    supporting_evidence: Mapping[str, Any]
    counter_evidence: Mapping[str, Any]
    partial_correspondence: Mapping[str, Any]
    warnings: List[str]
    recommended_next_action: str
    decision_trace: Mapping[str, Any]
    limitations: List[str]
    prohibited_claims: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
