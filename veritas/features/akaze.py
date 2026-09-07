"""AKAZE feature extraction as an independent binary-descriptor evidence path."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..schemas import EVIDENCE_FAMILY_AKAZE, FeatureEvidence


class AkazeDetector:
    """Detect AKAZE keypoints and MLDB binary descriptors."""

    def __init__(self, threshold: float = 0.001) -> None:
        self.detector = cv2.AKAZE_create(threshold=threshold)

    def detect(self, image: np.ndarray, descriptor_offset: int = 0) -> Tuple[List[FeatureEvidence], np.ndarray]:
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, 61), dtype=np.uint8)
        return [
            FeatureEvidence(EVIDENCE_FAMILY_AKAZE, float(kp.pt[0]), float(kp.pt[1]), float(kp.size),
                            float(kp.angle), float(kp.response), descriptor_offset + index)
            for index, kp in enumerate(keypoints)
        ], descriptors.astype(np.uint8, copy=False)
