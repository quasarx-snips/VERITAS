"""Contract declarations for the VERITAS evidence chain.

INCOMPLETE MODULE — declared contracts; no implementation yet.

This module intentionally declares contracts before any algorithm exists so
that migration commits can implement against stable interfaces. No function,
dataclass, or fake default is shipped before its migration commit lands
(see the NO FAKE IMPLEMENTATION rule in the project protocol).

Planned contracts (in migration order):
-------------------
- ``FeatureEvidence``    : one detected local feature — coordinates, scale,
                           orientation, confidence, descriptor index,
                           evidence family.
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