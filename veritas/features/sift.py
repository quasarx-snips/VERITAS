"""SIFT feature detection and description."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ..schemas import EVIDENCE_FAMILY_SIFT, FeatureEvidence

ImageArray = np.ndarray


class SiftDetector:
    """Detects and describes SIFT features, with optional RootSIFT output."""

    def __init__(
        self,
        n_features: int = 6000,
        contrast_threshold: float = 0.01,
        edge_threshold: float = 15.0,
        root_sift: bool = False,
    ) -> None:
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )
        self.root_sift = root_sift

    def _normalize_descriptors(self, descriptors: Optional[np.ndarray]) -> np.ndarray:
        """Apply RootSIFT's L1 + square-root normalization when enabled."""
        if descriptors is None:
            return np.empty((0, 128), dtype=np.float32)
        descriptors = descriptors.astype(np.float32, copy=False)
        if not self.root_sift or not len(descriptors):
            return descriptors
        return np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12))

    def detect(
        self, enhanced_img: ImageArray, descriptor_offset: int = 0
    ) -> Tuple[List[FeatureEvidence], np.ndarray]:
        """Detect SIFT features and return ``(evidence_records, descriptors)``."""
        kps, descs = self.sift.detectAndCompute(enhanced_img, None)
        descs = self._normalize_descriptors(descs)
        return [
            FeatureEvidence(
                evidence_family=EVIDENCE_FAMILY_SIFT,
                x=float(kp.pt[0]),
                y=float(kp.pt[1]),
                scale=float(kp.size),
                orientation=float(kp.angle),
                confidence=float(kp.response),
                descriptor_index=descriptor_offset + i,
            )
            for i, kp in enumerate(kps)
        ], descs

    def describe_features(
        self, enhanced_img: ImageArray, features: List[FeatureEvidence]
    ) -> np.ndarray:
        """Compute one SIFT descriptor for every supplied feature record.

        Records too close to an image border may have no valid descriptor and
        retain ``descriptor_index=None``.
        """
        for feature in features:
            feature.descriptor_index = None
        if not features:
            return np.empty((0, 128), dtype=np.float32)
        keypoints = []
        for index, feature in enumerate(features):
            size = max(3.0, float(feature.scale) if feature.scale else 8.0)
            angle = float(feature.orientation) if feature.orientation is not None else -1.0
            keypoints.append(
                cv2.KeyPoint(
                    float(feature.x),
                    float(feature.y),
                    size,
                    angle,
                    float(feature.confidence),
                    0,
                    index,
                )
            )
        described_keypoints, descriptors = self.sift.compute(enhanced_img, keypoints)
        if descriptors is None:
            return np.empty((0, 128), dtype=np.float32)
        descriptors = self._normalize_descriptors(descriptors)
        for descriptor_index, keypoint in enumerate(described_keypoints):
            original_index = keypoint.class_id
            if 0 <= original_index < len(features):
                features[original_index].descriptor_index = descriptor_index
        return descriptors