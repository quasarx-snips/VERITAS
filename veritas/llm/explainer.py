"""Build zero-pixel explanation input; this module is not an LLM runtime."""

from typing import Dict, List

from veritas.audit.provenance import json_safe
from veritas.gate.schema import GateResult
from veritas.schemas import VerificationEvidence
from veritas.verdict.schema import VerdictResult
from veritas.verification.partial import PartialCorrespondenceResult

from .schema import ExplanationPayload


_REASON_TEXT: Dict[str, str] = {
    "AFFINE_CERTIFIED": "The affine certificate was verified.",
    "AFFINE_FAILED": "The affine certificate could not be verified.",
    "AFFINE_CERTIFICATE_FAILED": "The affine certificate could not be verified.",
    "DISTRIBUTED_SUPPORT": "Support is distributed across the image.",
    "DETECTOR_AGREEMENT": "Detector evidence agrees.",
    "DETECTOR_DISAGREEMENT": "Detector evidence is inconsistent.",
    "PARTIAL_OVERLAP": "Only partial correspondence support is available.",
    "PARTIAL_CORRESPONDENCE_SUPPORT": "Local regions provide partial correspondence support.",
    "FULL_CORRESPONDENCE_SUPPORT": "All evaluated regions provide correspondence support.",
    "INSUFFICIENT_CORRESPONDENCE": "Correspondence support is insufficient.",
    "LOW_SPATIAL_COVERAGE": "Spatial support covers too little of the image.",
    "LOW_SPATIAL_ENTROPY": "Spatial support is unevenly distributed.",
    "HIGH_RESIDUAL": "Reprojection residuals are high.",
    "HIGH_SPATIAL_CONCENTRATION": "Support is spatially concentrated.",
    "HIGH_COUNTER_EVIDENCE": "Counter-evidence requires a restrictive action.",
}

_NEXT_ACTION = {
    "ALLOW": "Proceed with the authorized downstream workflow.",
    "RESTRICT": "Use only supported regions or apply the configured restricted workflow.",
    "HUMAN_REVIEW": "Route this pair to a human reviewer before downstream action.",
    "BLOCK": "Do not run downstream action; obtain a new or better-supported image pair.",
}

_PROHIBITED = [
    "Do not claim geographic identity without evidence.",
    "Do not call heuristic scores probabilities.",
    "Do not claim semantic change from correspondence evidence.",
    "Do not override the deterministic gate.",
]


def build_explanation_payload(
    evidence: VerificationEvidence,
    partial_correspondence: PartialCorrespondenceResult,
    verdict: VerdictResult,
    gate: GateResult,
    decision_trace: Dict[str, object],
) -> ExplanationPayload:
    """Create a deterministic explanation input without image pixels."""
    codes = sorted(set(gate.reasons or verdict.rationale_codes))
    key_reasons = [_REASON_TEXT.get(code, code.replace("_", " ").title()) for code in codes]
    warnings: List[str] = list(verdict.limitations)
    if gate.action.value != "ALLOW":
        warnings.append("The gate action is binding for downstream consumers.")
    return ExplanationPayload(
        schema_version="3C.1",
        task="Explain a deterministic VERITAS correspondence decision using supplied structured evidence only.",
        headline=f"VERITAS verdict: {verdict.verdict.value}; gate: {gate.action.value}.",
        verdict=verdict.verdict.value,
        gate_action=gate.action.value,
        key_reasons=key_reasons,
        supporting_evidence=json_safe(verdict.supporting_evidence),
        counter_evidence=json_safe(evidence.counter_evidence.to_dict()),
        partial_correspondence=json_safe(partial_correspondence.to_dict()),
        warnings=sorted(set(warnings)),
        recommended_next_action=_NEXT_ACTION[gate.action.value],
        decision_trace=json_safe(decision_trace),
        limitations=list(verdict.limitations),
        prohibited_claims=list(_PROHIBITED),
    )
