"""Safety-critical deterministic verdict gate.

This module deliberately accepts only structured evidence and verdict data.
It imports neither LLM nor TTS components and emits only action/reason codes.
"""

from typing import List, Optional

from veritas.schemas import VerificationEvidence
from veritas.verdict.schema import VerdictResult

from .schema import GatePolicy, GateResult


class DownstreamSafetyGate:
    """Map a verdict and structured evidence to a configured safe action."""

    def __init__(self, policy: Optional[GatePolicy] = None):
        self.policy = policy or GatePolicy()

    def decide(self, verdict_result: VerdictResult, evidence: VerificationEvidence) -> GateResult:
        verdict = verdict_result.verdict.value
        reasons: List[str] = list(verdict_result.rationale_codes)
        if evidence.geometry.certified:
            reasons.append("AFFINE_CERTIFIED")
        else:
            reasons.append("AFFINE_CERTIFICATE_FAILED")
        if evidence.spatial.coverage_ratio < 0.6:
            reasons.append("LOW_SPATIAL_COVERAGE")
        if evidence.counter_evidence.feature_disagreement.get("disagreement_score", 0.0) >= 0.2:
            reasons.append("DETECTOR_DISAGREEMENT")
        residuals = evidence.counter_evidence.residuals
        max_residual = residuals.get("max_reprojection_error", residuals.get("maximum", 0.0))
        if max_residual is not None and max_residual > 2.0:
            reasons.append("HIGH_RESIDUAL")
        if evidence.counter_evidence.spatial_concentration.get("concentrated", False):
            reasons.append("HIGH_COUNTER_EVIDENCE")
        return GateResult(
            action=self.policy.action_for(verdict),
            reasons=sorted(set(reasons)),
            verdict=verdict,
            policy=self.policy.to_dict(),
        )
