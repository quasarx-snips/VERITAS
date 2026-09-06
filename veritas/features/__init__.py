"""Independent feature-evidence families.

P0.2: migrated — SIFT + RootSIFT (``SiftDetector``) and feature persistence
(``FeatureStore``); records use the ``FeatureEvidence`` contract from
``veritas.schemas``. ORB and AKAZE arrive in P1 as separate evidence families
(independent implementation paths with different descriptor failure modes).
Ridge/texture structure detectors are deferred and will be re-framed as
complementary gradient/line evidence, not terrain landmarks.
"""

from .sift import SiftDetector
from .store import FeatureStore

__all__ = ["SiftDetector", "FeatureStore"]