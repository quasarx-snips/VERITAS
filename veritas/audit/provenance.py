"""Portable provenance helpers for VERITAS audit reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4


def portable_identifier(value: Any, fallback: str) -> str:
    """Return a logical identifier without embedding a personal absolute path."""
    if isinstance(value, (str, Path)):
        return Path(value).name
    return fallback


def json_safe(value: Any) -> Any:
    """Convert numpy/OpenCV-like values recursively into JSON primitives."""
    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return json_safe(value.item())
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class Provenance:
    schema_version: str
    run_id: str
    timestamp: str
    input_identifiers: Mapping[str, str]
    image_dimensions: Mapping[str, Optional[Dict[str, int]]]
    configurations: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source_input: Any,
        reference_input: Any,
        source_shape: Optional[tuple] = None,
        reference_shape: Optional[tuple] = None,
        configurations: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> "Provenance":
        def dimensions(shape: Optional[tuple]) -> Optional[Dict[str, int]]:
            if shape is None or len(shape) < 2:
                return None
            return {"height": int(shape[0]), "width": int(shape[1])}
        return cls(
            schema_version="3B.1",
            run_id=run_id or str(uuid4()),
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            input_identifiers={
                "source": portable_identifier(source_input, "source_image"),
                "reference": portable_identifier(reference_input, "reference_image"),
            },
            image_dimensions={"source": dimensions(source_shape), "reference": dimensions(reference_shape)},
            configurations=json_safe(configurations or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))
