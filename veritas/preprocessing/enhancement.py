"""VERITAS preprocessing — CLAHE contrast enhancement.

Enhancement responsibility: apply deterministic contrast-limited adaptive
histogram equalization (CLAHE) to a normalized ``uint8`` grayscale image.

Configuration is explicit — ``ClaheConfig`` — and fully deterministic: the same
config applied to the same input yields the same output. CLAHE is used for
contrast normalization of arbitrary image pairs; it is a local contrast
enhancement, not a claim about scene properties.

Dependencies: opencv-python, numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np

ImageArray = np.ndarray


@dataclass(frozen=True)
class ClaheConfig:
    """Deterministic CLAHE configuration.

    Attributes:
        clip_limit: contrast clipping limit.
        tile_grid_size: (rows, cols) tile grid for adaptive equalization.
    """

    clip_limit: float = 2.5
    tile_grid_size: Tuple[int, int] = (8, 8)


class ClaheEnhancer:
    """Applies CLAHE contrast enhancement to uint8 grayscale images."""

    def __init__(self, config: ClaheConfig | None = None) -> None:
        self.config = config or ClaheConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=self.config.tile_grid_size,
        )

    def enhance(self, image: Any) -> np.ndarray:
        """Apply CLAHE.

        Args:
            image: ``uint8`` grayscale array (use ``normalize`` first).

        Returns:
            Enhanced ``uint8`` grayscale array of the same shape.
        """
        array = np.asarray(image)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("CLAHE enhancement requires a non-empty grayscale array")
        return self.clahe.apply(array)


def enhance(image: Any, config: ClaheConfig | None = None) -> np.ndarray:
    """Apply CLAHE enhancement with an explicit, deterministic configuration."""
    return ClaheEnhancer(config).enhance(image)