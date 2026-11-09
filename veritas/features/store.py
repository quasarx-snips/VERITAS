"""VERITAS features — persistence for feature records and descriptors.

P0.2 migration: adapted from the reference ``FeatureStore`` (JSON feature
records, NPY descriptor arrays). No domain-specific metadata is assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..schemas import FeatureEvidence


class FeatureStore:
    """Saves and loads feature records and their descriptor arrays."""

    @staticmethod
    def save_features(
        path: Union[str, Path],
        features: List[FeatureEvidence],
        image_shape: Tuple[int, int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "image_shape": list(image_shape),
                    "feature_count": len(features),
                    "metadata": metadata or {},
                    "features": [f.to_dict() for f in features],
                },
                indent=2,
            )
        )

    @staticmethod
    def save_descriptors(path: Union[str, Path], descriptors: np.ndarray) -> None:
        np.save(str(path), descriptors)