"""VERITAS policy configuration.

This module holds the few architecture-level policies that must remain stable
across migration commits. It imports no heavy dependencies so that every
subpackage can pull policy constants cheaply.

Geometry certification policy
-----------------------------
VERITAS certifies correspondences with an AFFINE-ONLY geometric model.

- No automatic model selection (``AUTO``) is used for certification.
- No silent fallback to homography is permitted.
- Homography is not an accepted VERITAS certification model.
"""

from typing import Final

#: The only geometric model accepted for VERITAS correspondence certification.
CERTIFICATION_MODELS: Final[tuple[str, ...]] = ("affine",)

#: Baseline verification thresholds. These are tuned defaults used by the
#: affine verifier once the geometry migration lands; they are policy inputs,
#: not claims about intrinsic scene properties.
VERIFICATION_DEFAULTS: Final[dict] = {
    "reprojection_threshold_px": 3.0,
    "ransac_confidence": 0.99,
    "ransac_max_iterations": 2000,
    "min_inliers": 4,
}