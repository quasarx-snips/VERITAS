"""Evidence bundling; intentionally no verdict or calibrated probability."""
from veritas.schemas import VerificationEvidence

def fuse_evidence(features, matches, geometry, spatial, quorum, counter_evidence):
    """Return traceable evidence inputs for a later Phase 3 classifier."""
    return VerificationEvidence(features, matches, geometry, spatial, quorum, counter_evidence)
