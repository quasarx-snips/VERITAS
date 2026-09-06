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
from typing import Any, Dict, Optional

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