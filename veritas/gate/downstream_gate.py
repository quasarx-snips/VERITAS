"""Safety-critical deterministic verdict gate.

This module deliberately accepts only structured evidence and verdict data.
It imports neither LLM nor TTS components and emits only action/reason codes.
"""

from __future__ import annotations

from typing import List, Optional

from veritas.schemas import VerificationEvidence
from veritas.verdict.schema import VerdictResult

from .schema import GatePolicy, GateResult

#: Spatial coverage ratio below which the gate flags low coverage.
LOW_SPATIAL_COVERAGE_RATIO: float = 0.6
#: Detector disagreement score at or above which the gate flags disagreement.
DETECTOR_DISAGREEMENT_THRESHOLD: float = 0.2
#: Maximum reprojection error (pixel units) before the gate flags a high residual.
MAX_RESIDUAL_THRESHOLD: float = 2.0


class DownstreamSafetyGate:
    """Map a verdict and structured evidence to a configured safe action."""

    def __init__(self, policy: Optional[GatePolicy] = None) -> None:
        self.policy = policy or GatePolicy()

    def decide(
        self, verdict_result: VerdictResult, evidence: VerificationEvidence
    ) -> GateResult:
        """Map a verdict and its structured evidence to a gate action.

        Args:
            verdict_result: The classified verdict result.
            evidence: The fused verification evidence bundle.

        Returns:
            A :class:`GateResult` with an action, sorted reason codes, verdict
            and the applied policy.
        """
        verdict = verdict_result.verdict.value
        reasons: List[str] = list(verdict_result.rationale_codes)
        if evidence.geometry.certified:
            reasons.append("AFFINE_CERTIFIED")
        else:
            reasons.append("AFFINE_CERTIFICATE_FAILED")
        if evidence.spatial.coverage_ratio < LOW_SPATIAL_COVERAGE_RATIO:
            reasons.append("LOW_SPATIAL_COVERAGE")
        if (
            evidence.counter_evidence.feature_disagreement.get("disagreement_score", 0.0)
            >= DETECTOR_DISAGREEMENT_THRESHOLD
        ):
            reasons.append("DETECTOR_DISAGREEMENT")
        residuals = evidence.counter_evidence.residuals
        max_residual = residuals.get("max_reprojection_error", residuals.get("maximum", 0.0))
        if max_residual is not None and max_residual > MAX_RESIDUAL_THRESHOLD:
            reasons.append("HIGH_RESIDUAL")
        if evidence.counter_evidence.spatial_concentration.get("concentrated", False):
            reasons.append("HIGH_COUNTER_EVIDENCE")
        return GateResult(
            action=self.policy.action_for(verdict),
            reasons=sorted(set(reasons)),
            verdict=verdict,
            policy=self.policy.to_dict(),
        )
