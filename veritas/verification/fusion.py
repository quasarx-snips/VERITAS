"""Evidence bundling; intentionally no verdict or calibrated probability."""

from __future__ import annotations

from typing import Any, Mapping

from veritas.schemas import VerificationEvidence


def fuse_evidence(
    features: Mapping[str, Any],
    matches: Mapping[str, Any],
    geometry: Any,
    spatial: Any,
    quorum: Any,
    counter_evidence: Any,
) -> VerificationEvidence:
    """Bundle traceable evidence inputs for a later Phase 3 classifier.

    Args:
        features: Per-detector feature evidence.
        matches: Per-detector match evidence.
        geometry: Geometric certificate evidence.
        spatial: Spatial coverage/entropy evidence.
        quorum: Detector quorum evidence.
        counter_evidence: Negative counter-evidence bundle.

    Returns:
        A :class:`~veritas.schemas.VerificationEvidence` combining all inputs.
    """
    return VerificationEvidence(
        features, matches, geometry, spatial, quorum, counter_evidence
    )
