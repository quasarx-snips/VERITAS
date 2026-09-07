"""Serializable audit-report schema and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .provenance import json_safe


@dataclass(frozen=True)
class AuditReport:
    schema_version: str
    run_id: str
    input: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    detectors: Mapping[str, Any]
    matching: Mapping[str, Any]
    geometry: Mapping[str, Any]
    spatial: Mapping[str, Any]
    counter_evidence: Mapping[str, Any]
    partial_correspondence: Mapping[str, Any]
    fused_evidence: Mapping[str, Any]
    verdict: Mapping[str, Any]
    gate: Mapping[str, Any]
    provenance: Mapping[str, Any]
    decision_trace: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(self.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditReport":
        cls.validate(data)
        return cls(**dict(data))

    @classmethod
    def from_json(cls, value: str) -> "AuditReport":
        return cls.from_dict(json.loads(value))

    @staticmethod
    def validate(data: Mapping[str, Any]) -> None:
        required = set(AuditReport.__dataclass_fields__)
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Audit report missing required fields: {sorted(missing)}")
        if not data["schema_version"] or not data["run_id"]:
            raise ValueError("Audit report requires schema_version and run_id")
