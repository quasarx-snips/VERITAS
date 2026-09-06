"""Contract declarations for the VERITAS evidence chain.

Implemented contracts are marked below; the rest remain declarations until
their migration commits land (see the NO FAKE IMPLEMENTATION rule in the
project protocol).

Implemented:
-----------
- ``FeatureEvidence``: one detected local feature — coordinates, scale,
                       orientation, confidence, descriptor index, and the
                       evidence family that produced it (P0.2, migrated from
                       the reference ``TerrainFeature`` record and renamed).

Declared (not yet implemented):
-----------
- ``CorrespondenceSet``  : accepted descriptor correspondences with index
                           records, before any geometric proof.
- ``AffineCertificate``  : affine-only geometric verification output —
                           transformation, inlier mask, diagnostics, error
                           statistics.
- ``SpatialEvidence``    : coverage / density / entropy over verified
                           correspondences.
- ``CounterEvidence``    : checks that argue against a verified correspondence.
- ``VerdictReport``      : fused evidence, verdict class, and the audit trail
                           that produced it.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

#: Evidence families are treated as complementary evidence families and
#: independent implementation paths with different descriptor failure modes —
#: never as universally independent statistical measurements.
EVIDENCE_FAMILY_SIFT = "sift"
EVIDENCE_FAMILY_ORB = "orb"
EVIDENCE_FAMILY_AKAZE = "akaze"


@dataclass
class FeatureEvidence:
    """One detected local feature with an optional descriptor association.

    ``evidence_family`` identifies which detector produced the feature
    (e.g. ``"sift"``, ``"orb"``, ``"akaze"``). ``descriptor_index`` links the
    record to a row in the descriptor array produced by the same detector;
    records without a descriptor keep ``descriptor_index=None``.
    """

    evidence_family: str
    x: float
    y: float
    scale: Optional[float] = None
    orientation: Optional[float] = None
    confidence: float = 0.0
    descriptor_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectorEvidence:
    """Computed feature-extraction evidence for one detector family."""

    detector: str
    keypoint_count: int
    descriptor_count: int
    descriptor_type: str
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchEvidence:
    """Actual matching counts and filter diagnostics for one detector."""

    detector: str
    candidate_count: int
    filtered_count: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "diagnostics": dict(self.diagnostics)}


@dataclass(frozen=True)
class GeometryEvidence:
    """AFFINE-only certificate facts, represented without a confidence score."""

    model: str
    certified: bool
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    residuals: Mapping[str, float]
    affine_matrix: Optional[List[List[float]]]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "residuals": dict(self.residuals)}


@dataclass(frozen=True)
class SpatialEvidence:
    """Grid occupancy and Shannon-distribution measurements."""

    grid_rows: int
    grid_columns: int
    occupied_cells: int
    coverage_ratio: float
    normalized_entropy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuorumEvidence:
    """Detector support states; unavailable evidence is not disagreement."""

    detector_results: Mapping[str, str]
    available_detectors: List[str]
    supporting_detectors: List[str]
    disagreeing_detectors: List[str]
    quorum_strength: float

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "detector_results": dict(self.detector_results)}


@dataclass(frozen=True)
class CounterEvidence:
    """Inspectable negative evidence, never a verdict."""

    residuals: Mapping[str, Any]
    feature_disagreement: Mapping[str, Any]
    spatial_concentration: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residuals": dict(self.residuals),
            "feature_disagreement": dict(self.feature_disagreement),
            "spatial_concentration": dict(self.spatial_concentration),
        }


@dataclass(frozen=True)
class VerificationEvidence:
    """Phase 2 evidence bundle for a later decision layer."""

    features: Mapping[str, DetectorEvidence]
    matches: Mapping[str, MatchEvidence]
    geometry: GeometryEvidence
    spatial: SpatialEvidence
    quorum: QuorumEvidence
    counter_evidence: CounterEvidence
    evidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": {name: value.to_dict() for name, value in self.features.items()},
            "matches": {name: value.to_dict() for name, value in self.matches.items()},
            "geometry": self.geometry.to_dict(),
            "spatial": self.spatial.to_dict(),
            "quorum": self.quorum.to_dict(),
            "counter_evidence": self.counter_evidence.to_dict(),
            "evidence_score": self.evidence_score,
        }