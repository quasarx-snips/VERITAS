"""Deterministic VERITAS verdict classifier.

This module implements the core logic for classifying the correspondence
between two images into one of the VERITAS verdict categories: STRONG, PARTIAL,
WEAK, or NONE. It consumes structured evidence from Phase 1 and Phase 2
of the VERITAS pipeline and produces a `VerdictResult` that includes the
verdict, rationale codes, supporting evidence, counter-evidence, and
decision trace.

The classifier is deterministic and relies on a set of configurable thresholds
defined in `veritas.verdict.thresholds`. These thresholds are provisional
and heuristic, pending future calibration.

Key principles:
- **Deterministic Classification**: The verdict is derived from a clear set
  of rules and thresholds, not probabilistic models.
- **Explainability**: Each verdict is accompanied by machine-readable
  rationale codes explaining why that verdict was reached.
- **Structured Output**: The `VerdictResult` provides a comprehensive
  summary of the decision, including the evidence that led to it.
- **Separation of Concerns**: This classifier judges correspondence
  trustworthiness, not semantic changes in the scene.
"""

from typing import List, Dict, Any, Optional

from veritas.schemas import VerificationEvidence
from veritas.verdict.schema import VerdictResult, Verdict
from veritas.verdict.thresholds import VerdictThresholds


class VerdictClassifier:
    """Classifies VERITAS evidence into a definitive verdict."""

    def __init__(self, thresholds: VerdictThresholds):
        self.thresholds = thresholds

    def classify(self, evidence: VerificationEvidence, partial_correspondence_status: str) -> VerdictResult:
        """Classifies the given evidence into a VERITAS verdict.

        Args:
            evidence: The structured `VerificationEvidence` from Phase 2.
            partial_correspondence_status: The status from partial correspondence analysis (FULL, PARTIAL, INSUFFICIENT).

        Returns:
            A `VerdictResult` object containing the verdict and its rationale.
        """
        verdict = Verdict.NONE
        rationale_codes: List[str] = []
        decision_trace: List[Dict[str, Any]] = []
        confidence_status: Dict[str, Any] = {}
        limitations: List[str] = []

        # --- Initial Checks and Early Exits ---
        if not evidence.geometry.certified:
            verdict = Verdict.NONE
            rationale_codes.append("AFFINE_FAILED")
            decision_trace.append({"step": "affine_certification", "result": "failed"})
            limitations.append("Affine transformation could not be certified.")
            return self._build_result(
                verdict, rationale_codes, evidence, confidence_status, limitations, decision_trace
            )

        rationale_codes.append("AFFINE_CERTIFIED")
        decision_trace.append({"step": "affine_certification", "result": "certified"})

        # --- Evaluate Core Evidence ---
        inlier_ratio = evidence.geometry.inlier_ratio
        inlier_count = evidence.geometry.inlier_count
        spatial_coverage = evidence.spatial.coverage_ratio
        normalized_entropy = evidence.spatial.normalized_entropy
        quorum_strength = evidence.quorum.quorum_strength
        max_residual = evidence.counter_evidence.residuals.get("max_reprojection_error", 0.0)
        mean_residual = evidence.counter_evidence.residuals.get("mean_reprojection_error", 0.0)
        spatial_concentration_score = evidence.counter_evidence.spatial_concentration.get("concentration_score", 0.0)
        detector_disagreement_score = evidence.counter_evidence.feature_disagreement.get("disagreement_score", 0.0)

        # --- Decision Logic ---

        # Strong Verdict Conditions
        if (inlier_ratio >= self.thresholds.strong_inlier_ratio and
                inlier_count >= self.thresholds.strong_inlier_count and
                spatial_coverage >= self.thresholds.strong_spatial_coverage and
                (normalized_entropy is None or normalized_entropy >= self.thresholds.strong_spatial_entropy) and
                quorum_strength >= self.thresholds.strong_quorum_strength and
                max_residual <= self.thresholds.strong_max_residual and
                mean_residual <= self.thresholds.strong_mean_residual and
                spatial_concentration_score < self.thresholds.strong_spatial_concentration_threshold and
                detector_disagreement_score < self.thresholds.strong_detector_disagreement_threshold and
                partial_correspondence_status == "FULL"):
            verdict = Verdict.STRONG
            rationale_codes.extend([
                "DISTRIBUTED_SUPPORT",
                "DETECTOR_AGREEMENT",
                "FULL_CORRESPONDENCE_SUPPORT"
            ])
            decision_trace.append({"step": "strong_conditions", "result": "met"})

        # Partial Verdict Conditions (if not STRONG)
        elif (inlier_ratio >= self.thresholds.partial_inlier_ratio and
              inlier_count >= self.thresholds.partial_inlier_count and
              spatial_coverage >= self.thresholds.partial_spatial_coverage and
              (normalized_entropy is None or normalized_entropy >= self.thresholds.partial_spatial_entropy) and
              quorum_strength >= self.thresholds.partial_quorum_strength and
              partial_correspondence_status in ["FULL", "PARTIAL"]):
            verdict = Verdict.PARTIAL
            rationale_codes.append("PARTIAL_OVERLAP")
            if partial_correspondence_status == "FULL":
                rationale_codes.append("FULL_CORRESPONDENCE_SUPPORT")
            else:
                rationale_codes.append("PARTIAL_CORRESPONDENCE_SUPPORT")
            decision_trace.append({"step": "partial_conditions", "result": "met"})

        # Weak Verdict Conditions (if not STRONG or PARTIAL)
        elif (inlier_count >= self.thresholds.weak_inlier_count and
              inlier_ratio >= self.thresholds.weak_inlier_ratio and
              partial_correspondence_status in ["FULL", "PARTIAL", "INSUFFICIENT"]):
            verdict = Verdict.WEAK
            rationale_codes.append("INSUFFICIENT_CORRESPONDENCE")
            decision_trace.append({"step": "weak_conditions", "result": "met"})

        # Default to NONE if no other conditions met
        else:
            verdict = Verdict.NONE
            rationale_codes.append("INSUFFICIENT_CORRESPONDENCE")
            decision_trace.append({"step": "none_conditions", "result": "default"})

        # --- Add Counter-Evidence Rationale ---
        if max_residual > self.thresholds.strong_max_residual or mean_residual > self.thresholds.strong_mean_residual:
            rationale_codes.append("HIGH_RESIDUAL")
        if spatial_coverage < self.thresholds.strong_spatial_coverage:
            rationale_codes.append("LOW_SPATIAL_COVERAGE")
        if normalized_entropy is not None and normalized_entropy < self.thresholds.strong_spatial_entropy:
            rationale_codes.append("LOW_SPATIAL_ENTROPY")
        if detector_disagreement_score >= self.thresholds.strong_detector_disagreement_threshold:
            rationale_codes.append("DETECTOR_DISAGREEMENT")
        if spatial_concentration_score >= self.thresholds.strong_spatial_concentration_threshold:
            rationale_codes.append("HIGH_SPATIAL_CONCENTRATION")

        # --- Populate Confidence Status (Heuristic Scores) ---
        confidence_status["inlier_ratio_heuristic"] = inlier_ratio
        confidence_status["inlier_count_heuristic"] = inlier_count
        confidence_status["spatial_coverage_heuristic"] = spatial_coverage
        confidence_status["spatial_entropy_heuristic"] = normalized_entropy
        confidence_status["quorum_strength_heuristic"] = quorum_strength
        confidence_status["max_residual_heuristic"] = max_residual
        confidence_status["mean_residual_heuristic"] = mean_residual
        confidence_status["spatial_concentration_heuristic"] = spatial_concentration_score
        confidence_status["detector_disagreement_heuristic"] = detector_disagreement_score
        confidence_status["partial_correspondence_status"] = partial_correspondence_status

        return self._build_result(
            verdict, rationale_codes, evidence, confidence_status, limitations, decision_trace
        )

    def _build_result(
        self,
        verdict: Verdict,
        rationale_codes: List[str],
        evidence: VerificationEvidence,
        confidence_status: Dict[str, Any],
        limitations: List[str],
        decision_trace: List[Dict[str, Any]],
    ) -> VerdictResult:
        """Helper to construct the VerdictResult."""
        return VerdictResult(
            verdict=verdict,
            rationale_codes=sorted(list(set(rationale_codes))),  # Remove duplicates and sort
            supporting_evidence={
                "geometry": evidence.geometry.to_dict(),
                "spatial": evidence.spatial.to_dict(),
                "quorum": evidence.quorum.to_dict(),
            },
            counter_evidence=evidence.counter_evidence.to_dict(),
            decision_trace=decision_trace,
            confidence_status=confidence_status,
            limitations=limitations,
        )