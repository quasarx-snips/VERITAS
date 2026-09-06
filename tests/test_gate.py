"""Tests for the deterministic downstream safety gate."""

from veritas.gate import DownstreamSafetyGate, GateAction, GatePolicy
from veritas.schemas import CounterEvidence, GeometryEvidence, QuorumEvidence, SpatialEvidence, VerificationEvidence
from veritas.verdict.schema import Verdict, VerdictResult


def evidence(*, certified=True):
    return VerificationEvidence(
        features={}, matches={},
        geometry=GeometryEvidence("affine", certified, 8, 8, 1.0, {}, None),
        spatial=SpatialEvidence(4, 4, 16, 1.0, 1.0),
        quorum=QuorumEvidence({}, [], [], [], 1.0),
        counter_evidence=CounterEvidence({}, {}, {}),
    )


def verdict(value):
    return VerdictResult(Verdict(value), rationale_codes=["TEST_EVIDENCE"])


def test_default_verdict_action_mapping_and_reasons():
    gate = DownstreamSafetyGate()
    assert gate.decide(verdict("STRONG"), evidence()).action == GateAction.ALLOW
    assert gate.decide(verdict("WEAK"), evidence()).action == GateAction.HUMAN_REVIEW
    assert gate.decide(verdict("NONE"), evidence(certified=False)).action == GateAction.BLOCK
    result = gate.decide(verdict("NONE"), evidence(certified=False))
    assert "AFFINE_CERTIFICATE_FAILED" in result.reasons
    assert "TEST_EVIDENCE" in result.reasons


def test_partial_mapping_is_configurable_and_restrictive():
    policy = GatePolicy({"STRONG": "ALLOW", "PARTIAL": "HUMAN_REVIEW", "WEAK": "HUMAN_REVIEW", "NONE": "BLOCK"})
    assert DownstreamSafetyGate(policy).decide(verdict("PARTIAL"), evidence()).action == GateAction.HUMAN_REVIEW


def test_gate_has_no_llm_or_tts_requirement():
    result = DownstreamSafetyGate().decide(verdict("STRONG"), evidence())
    assert result.reasons
