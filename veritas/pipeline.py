"""Phase 1 image-pair verification orchestration.

The pipeline intentionally ends at raw affine and spatial evidence. It does
not fuse evidence, make a verdict, or invoke downstream gating.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from .features import AkazeDetector, OrbDetector, SiftDetector, assess_quorum
from .counter_evidence import summarize_feature_disagreement, summarize_residuals, summarize_spatial_concentration
from .geometry import AffineCertificate, AffineVerificationConfig, verify_affine_from_correspondences
from .matching import MatchResult, match_feature_sets
from .preprocessing import ImagePreprocessor, PreprocessedImage
from .spatial import CoverageConfig, calculate_spatial_coverage, calculate_spatial_entropy
from .schemas import CounterEvidence, DetectorEvidence, MatchEvidence, SpatialEvidence, VerificationEvidence
from .verification import evaluate_registration, fuse_evidence, geometry_evidence
from .verification.partial import analyze_partial_correspondence, PartialCorrespondenceConfig, PartialCorrespondenceResult
from .verdict.classifier import VerdictClassifier
from .verdict.schema import VerdictResult
from .verdict.thresholds import VerdictThresholds
from .gate import DownstreamSafetyGate, GatePolicy, GateResult
from .audit import AuditReport, Provenance, build_audit_report
from .audit.provenance import json_safe
from .llm import ExplanationPayload, build_explanation_payload


@dataclass(frozen=True)
class VeritasPipelineResult:
    """Comprehensive result of the VERITAS pipeline, including all phases."""

    evidence: VerificationEvidence
    partial_correspondence: PartialCorrespondenceResult
    verdict_result: VerdictResult
    gate_result: GateResult
    audit_report: AuditReport
    explanation_payload: ExplanationPayload

    @property
    def verdict(self) -> VerdictResult:
        """Canonical top-level verdict contract (legacy field remains available)."""
        return self.verdict_result

    @property
    def gate(self) -> GateResult:
        """Canonical top-level gate contract (legacy field remains available)."""
        return self.gate_result

    @property
    def audit(self) -> AuditReport:
        """Canonical top-level audit contract (legacy field remains available)."""
        return self.audit_report

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "evidence": self.evidence.to_dict(),
            "partial_correspondence": self.partial_correspondence.to_dict(),
            "verdict": self.verdict_result.to_dict(),
            "gate": self.gate_result.to_dict(),
            "audit": self.audit_report.to_dict(),
            "explanation_payload": self.explanation_payload.to_dict(),
        })


@dataclass
class Phase1VerificationResult:
    """Raw Phase 1 evidence produced for one source/reference image pair."""

    source: PreprocessedImage
    reference: PreprocessedImage
    source_feature_count: int
    reference_feature_count: int
    matches: MatchResult
    certificate: AffineCertificate
    metrics: Optional[Dict[str, Any]]
    spatial_coverage: Dict[str, Any]


def verify_image_pair(
    source_image: Any,
    reference_image: Any,
    *,
    preprocessor: Optional[ImagePreprocessor] = None,
    detector: Optional[SiftDetector] = None,
    matching_config: Optional[Dict[str, Any]] = None,
    geometry_config: Optional[AffineVerificationConfig] = None,
    coverage_config: Optional[CoverageConfig] = None,
) -> Phase1VerificationResult:
    """Run preprocessing through AFFINE-only spatial evidence.

    ``source_image`` coordinates are transformed into ``reference_image``
    coordinates. An invalid affine certificate remains raw evidence rather
    than being converted into a verdict or confidence score.
    """
    image_preprocessor = preprocessor or ImagePreprocessor()
    sift = detector or SiftDetector()
    source = image_preprocessor.process(source_image)
    reference = image_preprocessor.process(reference_image)
    source_features, source_descriptors = sift.detect(source.enhanced)
    reference_features, reference_descriptors = sift.detect(reference.enhanced)
    matches = match_feature_sets(
        (source_features, source_descriptors),
        (reference_features, reference_descriptors),
        matching_config,
    )
    certificate = verify_affine_from_correspondences(matches, geometry_config)
    coverage = calculate_spatial_coverage(
        certificate.source_inliers,
        source.enhanced.shape,
        coverage_config,
    )
    metrics = None
    if certificate.transformation is not None:
        metrics = evaluate_registration(
            matches.source_points,
            matches.reference_points,
            certificate.transformation,
            certificate.inlier_mask,
            image_shape=source.enhanced.shape,
        )
    return Phase1VerificationResult(
        source=source,
        reference=reference,
        source_feature_count=len(source_features),
        reference_feature_count=len(reference_features),
        matches=matches,
        certificate=certificate,
        metrics=metrics,
        spatial_coverage=coverage,
    )


def verify_evidence_pair(
    source_image: Any,
    reference_image: Any,
    *,
    matching_config=None,
    geometry_config=None,
    coverage_config: Optional[CoverageConfig] = None,
    partial_config: Optional[PartialCorrespondenceConfig] = None,
    thresholds_config_path: str = "configs/thresholds.yaml",
    gate_policy: Optional[GatePolicy] = None,
    run_id: Optional[str] = None,
) -> VeritasPipelineResult:
    """Run evidence, verdict, deterministic gate, and portable audit creation."""
    preprocessor = ImagePreprocessor()
    source, reference = preprocessor.process(source_image), preprocessor.process(reference_image)
    detectors = {"sift": SiftDetector(), "orb": OrbDetector(), "akaze": AkazeDetector()}
    feature_evidence, match_evidence, geometries, certificates, match_results = {}, {}, {}, {}, {}
    for name, detector in detectors.items():
        source_features, source_descriptors = detector.detect(source.enhanced)
        reference_features, reference_descriptors = detector.detect(reference.enhanced)
        feature_evidence[name] = DetectorEvidence(name, len(source_features), len(source_descriptors), str(source_descriptors.dtype), bool(len(source_features)))
        result = match_feature_sets((source_features, source_descriptors), (reference_features, reference_descriptors), matching_config)
        match_results[name] = result
        match_evidence[name] = MatchEvidence(name, result.candidate_count, result.accepted_count, result.filter_diagnostics, bool(len(source_features) and len(reference_features)))
        certificate = verify_affine_from_correspondences(result, geometry_config)
        certificates[name] = certificate
        geometries[name] = geometry_evidence(certificate)
    quorum = assess_quorum(geometries)
    # SIFT remains the Phase 1 certification channel; other certificates stay visible.
    primary = certificates["sift"]
    coverage = calculate_spatial_coverage(primary.source_inliers, source.enhanced.shape, coverage_config)
    entropy = calculate_spatial_entropy(primary.source_inliers, source.enhanced.shape, coverage_config)
    spatial = SpatialEvidence(*coverage["grid_shape"], coverage["occupied_cells"], coverage["coverage_ratio"], entropy["normalized_entropy"])
    residual_values = []
    if primary.transformation is not None:
        from .geometry import compute_reprojection_errors
        residual_values = compute_reprojection_errors(primary.source_inliers, primary.reference_inliers, primary.transformation)
    counter = CounterEvidence(summarize_residuals(residual_values, primary.threshold), summarize_feature_disagreement(quorum), summarize_spatial_concentration(coverage, entropy))

    # Phase 2 evidence bundle
    verification_evidence = fuse_evidence(feature_evidence, match_evidence, geometries["sift"], spatial, quorum, counter)

    # Phase 3: Partial Correspondence Analysis
    partial_correspondence_result = analyze_partial_correspondence(
        match_result=match_results["sift"],  # Use SIFT matches for primary analysis
        certificate=primary,
        image_shape=source.enhanced.shape,
        verification_evidence=verification_evidence,
        config=partial_config,
    )

    # Phase 3: Verdict Classification
    thresholds = VerdictThresholds.load_from_yaml(thresholds_config_path)
    classifier = VerdictClassifier(thresholds)
    verdict_result = classifier.classify(verification_evidence, partial_correspondence_result.status.value)
    gate_result = DownstreamSafetyGate(gate_policy).decide(verdict_result, verification_evidence)
    configurations = {
        "preprocessing": asdict(preprocessor.enhancer.config),
        "matching": matching_config or {},
        "geometry": asdict(geometry_config) if geometry_config is not None else {},
        "ransac": certificates["sift"].configuration,
        "spatial_grid": asdict(coverage_config) if coverage_config is not None else {"grid_shape": coverage["grid_shape"]},
        "partial_correspondence": asdict(partial_config) if partial_config is not None else asdict(PartialCorrespondenceConfig()),
        "verdict_thresholds": thresholds.to_dict(),
        "gate_policy": gate_result.policy,
    }
    provenance = Provenance.create(
        source_input=source_image,
        reference_input=reference_image,
        source_shape=source.enhanced.shape,
        reference_shape=reference.enhanced.shape,
        configurations=configurations,
        run_id=run_id,
    )
    audit_report = build_audit_report(
        verification_evidence,
        partial_correspondence_result,
        verdict_result,
        gate_result,
        provenance,
        preprocessing=asdict(preprocessor.enhancer.config),
    )
    explanation_payload = build_explanation_payload(
        verification_evidence,
        partial_correspondence_result,
        verdict_result,
        gate_result,
        audit_report.decision_trace,
    )

    return VeritasPipelineResult(
        evidence=verification_evidence,
        partial_correspondence=partial_correspondence_result,
        verdict_result=verdict_result,
        gate_result=gate_result,
        audit_report=audit_report,
        explanation_payload=explanation_payload,
    )
