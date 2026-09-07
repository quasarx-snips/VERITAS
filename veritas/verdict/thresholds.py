"""Provisional and heuristic thresholds for VERITAS verdict classification.

These thresholds are used by the `VerdictClassifier` to determine the
correspondence verdict (STRONG, PARTIAL, WEAK, NONE). They are currently
uncalibrated and serve as initial operational values.

Future Calibration Note:
------------------------
These thresholds are provisional. They will be replaced by statistically
calibrated values derived from extensive ground-truth datasets and
performance benchmarks in future phases. Do not treat these values as
final or statistically validated.
"""

from dataclasses import dataclass, field
from typing import Any, Dict
import yaml
import os


@dataclass
class VerdictThresholds:
    """Configurable thresholds for the VERITAS verdict classifier."""

    # --- STRONG Verdict Thresholds ---
    strong_inlier_ratio: float = field(default=0.7, metadata={"description": "Provisional: Minimum inlier ratio for STRONG verdict."})
    strong_inlier_count: int = field(default=50, metadata={"description": "Provisional: Minimum inlier count for STRONG verdict."})
    strong_spatial_coverage: float = field(default=0.6, metadata={"description": "Provisional: Minimum spatial coverage for STRONG verdict."})
    strong_spatial_entropy: float = field(default=0.7, metadata={"description": "Provisional: Minimum normalized spatial entropy for STRONG verdict."})
    strong_quorum_strength: float = field(default=0.8, metadata={"description": "Provisional: Minimum detector quorum strength for STRONG verdict."})
    strong_max_residual: float = field(default=3.0, metadata={"description": "Operational: Maximum reprojection error for STRONG verdict; aligned with the affine inlier certificate boundary."})
    strong_mean_residual: float = field(default=1.0, metadata={"description": "Provisional: Maximum mean reprojection error for STRONG verdict."})
    strong_spatial_concentration_threshold: float = field(default=0.3, metadata={"description": "Provisional: Maximum spatial concentration score for STRONG verdict (lower is better)."})
    strong_detector_disagreement_threshold: float = field(default=0.2, metadata={"description": "Provisional: Maximum detector disagreement score for STRONG verdict (lower is better)."})

    # --- PARTIAL Verdict Thresholds ---
    partial_inlier_ratio: float = field(default=0.4, metadata={"description": "Provisional: Minimum inlier ratio for PARTIAL verdict."})
    partial_inlier_count: int = field(default=20, metadata={"description": "Provisional: Minimum inlier count for PARTIAL verdict."})
    partial_spatial_coverage: float = field(default=0.3, metadata={"description": "Provisional: Minimum spatial coverage for PARTIAL verdict."})
    partial_spatial_entropy: float = field(default=0.4, metadata={"description": "Provisional: Minimum normalized spatial entropy for PARTIAL verdict."})
    partial_quorum_strength: float = field(default=0.5, metadata={"description": "Provisional: Minimum detector quorum strength for PARTIAL verdict."})
    partial_max_residual: float = field(default=5.0, metadata={"description": "Provisional: Maximum reprojection error for PARTIAL verdict."})
    partial_mean_residual: float = field(default=2.5, metadata={"description": "Provisional: Maximum mean reprojection error for PARTIAL verdict."})

    # --- WEAK Verdict Thresholds ---
    weak_inlier_ratio: float = field(default=0.1, metadata={"description": "Provisional: Minimum inlier ratio for WEAK verdict."})
    weak_inlier_count: int = field(default=5, metadata={"description": "Provisional: Minimum inlier count for WEAK verdict."})

    def to_dict(self) -> Dict[str, Any]:
        """Converts the thresholds to a dictionary."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerdictThresholds":
        """Creates a VerdictThresholds instance from a dictionary."""
        return cls(**data)

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "VerdictThresholds":
        """Loads thresholds from a YAML file."""
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)
