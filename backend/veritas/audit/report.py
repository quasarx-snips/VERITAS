"""Audit report construction from existing structured pipeline outputs."""

from typing import Any, Mapping, Optional

from veritas.gate.schema import GateResult
from veritas.schemas import VerificationEvidence
from veritas.verdict.schema import VerdictResult
from veritas.verification.partial import PartialCorrespondenceResult

from .provenance import Provenance, json_safe
from .schema import AuditReport


def build_audit_report(
    evidence: VerificationEvidence,
    partial_correspondence: PartialCorrespondenceResult,
    verdict: VerdictResult,
    gate: GateResult,
    provenance: Provenance,
    *,
    preprocessing: Optional[Mapping[str, Any]] = None,
) -> AuditReport:
    """Preserve all decision inputs in one JSON-safe, inspectable report."""
    evidence_data = json_safe(evidence.to_dict())
    partial_data = json_safe(partial_correspondence.to_dict())
    verdict_data = json_safe(verdict.to_dict())
    gate_data = json_safe(gate.to_dict())
    trace = {
        "input": dict(provenance.input_identifiers),
        "preprocessing": json_safe(preprocessing or {}),
        "detector_evidence": evidence_data["features"],
        "matching": evidence_data["matches"],
        "geometry": evidence_data["geometry"],
        "spatial_evidence": evidence_data["spatial"],
        "counter_evidence": evidence_data["counter_evidence"],
        "partial_correspondence": partial_data,
        "fused_evidence": evidence_data,
        "verdict": verdict_data,
        "gate": gate_data,
    }
    return AuditReport(
        schema_version=provenance.schema_version,
        run_id=provenance.run_id,
        input={"identifiers": dict(provenance.input_identifiers), "dimensions": dict(provenance.image_dimensions)},
        preprocessing=json_safe(preprocessing or {}),
        detectors=evidence_data["features"],
        matching=evidence_data["matches"],
        geometry=evidence_data["geometry"],
        spatial=evidence_data["spatial"],
        counter_evidence=evidence_data["counter_evidence"],
        partial_correspondence=partial_data,
        fused_evidence=evidence_data,
        verdict=verdict_data,
        gate=gate_data,
        provenance=provenance.to_dict(),
        decision_trace=trace,
    )
