"""ORB feature extraction as an independent binary-descriptor evidence path."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..schemas import EVIDENCE_FAMILY_ORB, FeatureEvidence


class OrbDetector:
    """Detect ORB keypoints and binary descriptors with explicit parameters."""

    def __init__(self, n_features: int = 2000, scale_factor: float = 1.2, n_levels: int = 8) -> None:
        self.detector = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)

    def detect(self, image: np.ndarray, descriptor_offset: int = 0) -> Tuple[List[FeatureEvidence], np.ndarray]:
        """Return ORB records and aligned ``uint8`` binary descriptors."""
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        return [
            FeatureEvidence(EVIDENCE_FAMILY_ORB, float(kp.pt[0]), float(kp.pt[1]), float(kp.size),
                            float(kp.angle), float(kp.response), descriptor_offset + index)
            for index, kp in enumerate(keypoints)
        ], descriptors.astype(np.uint8, copy=False)
