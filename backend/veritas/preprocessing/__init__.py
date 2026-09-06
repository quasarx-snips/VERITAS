"""VERITAS preprocessing — image normalization and CLAHE enhancement.

P0.1: migrated from the reference repository's preprocessing module. The
implementation is domain-agnostic (works for arbitrary grayscale or colour
image pairs) and split into clear responsibilities:

- ``normalization.py`` — grayscale/BGR/BGRA handling, finite-value handling,
  robust ``uint8`` conversion;
- ``enhancement.py`` — deterministic CLAHE configuration and application.

``ImagePreprocessor`` is the compact facade; ``process`` returns the small
structured ``PreprocessedImage`` result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from .enhancement import ClaheConfig, ClaheEnhancer, enhance
from .normalization import ImageArray, normalize, normalize_to_uint8, to_grayscale

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PreprocessedImage:
    """Structured result of the preprocessing step.

    Attributes:
        normalized: finite ``uint8`` grayscale image (input normalized).
        enhanced: CLAHE-enhanced ``uint8`` grayscale image, same shape.
        source_shape: shape of the input accepted by ``process``.
    """

    normalized: np.ndarray
    enhanced: np.ndarray
    source_shape: tuple

    @property
    def shape(self) -> tuple:
        """Shape of the normalized/enhanced grayscale output."""
        return self.normalized.shape


class ImagePreprocessor:
    """Normalizes and enhances arbitrary grayscale or colour images.

    ``process`` accepts either an image path or a raw ``np.ndarray`` and
    returns a :class:`PreprocessedImage`.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.enhancer = ClaheEnhancer(
            ClaheConfig(clip_limit=clip_limit, tile_grid_size=tuple(tile_grid_size))
        )

    @staticmethod
    def load(image_path: PathLike) -> np.ndarray:
        """Load a grayscale image from disk."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return img

    def normalize(self, image: PathLike | ImageArray) -> np.ndarray:
        """Normalize a path or array to a finite grayscale uint8 image."""
        if isinstance(image, (str, Path)):
            image = self.load(image)
        return normalize(image)

    def enhance(self, image: PathLike | ImageArray) -> np.ndarray:
        """Apply CLAHE to a normalized uint8 grayscale path or array."""
        if isinstance(image, (str, Path)):
            image = self.load(image)
        return self.enhancer.enhance(image)

    def process(self, image: PathLike | ImageArray) -> PreprocessedImage:
        """Normalize then enhance a path or array, returning a structured result."""
        if isinstance(image, (str, Path)):
            image = self.load(image)
        array = np.asarray(image)
        normalized = normalize(array)
        enhanced = self.enhancer.enhance(normalized)
        return PreprocessedImage(
            normalized=normalized,
            enhanced=enhanced,
            source_shape=array.shape,
        )


__all__ = [
    "ClaheConfig",
    "ClaheEnhancer",
    "ImageArray",
    "ImagePreprocessor",
    "PreprocessedImage",
    "enhance",
    "normalize",
    "normalize_to_uint8",
    "to_grayscale",
]