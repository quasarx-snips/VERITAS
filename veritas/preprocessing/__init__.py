"""VERITAS preprocessing — image loading, validation, normalization, enhancement.

P0.1: migrated from the reference repository's preprocessing module. The
implementation is domain-agnostic (works for arbitrary grayscale or colour
image pairs) and split into clear responsibilities:

- ``normalization.py`` — grayscale/BGR/BGRA handling, finite-value handling,
  robust ``uint8`` conversion;
- ``enhancement.py`` — deterministic CLAHE configuration and application.

The facade adds decode validation, EXIF orientation handling (OpenCV applies
orientation during decode), optional bounded downscaling, and deterministic
quality metrics used for preflight warnings. Preflight metrics are advisory:
they never change the verdict or gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np

from .enhancement import ClaheConfig, ClaheEnhancer, enhance
from .normalization import ImageArray, normalize, normalize_to_uint8, to_grayscale

PathLike = Union[str, Path]

# 16 MP keeps full-resolution aerial frames workable while bounding memory.
DEFAULT_MAX_PIXELS = 16_000_000
DEFAULT_MIN_SIDE = 32


@dataclass(frozen=True)
class ImageQuality:
    """Advisory quality measurements for one input image."""

    width: int
    height: int
    megapixels: float
    blur_score: float
    mean_intensity: float
    intensity_std: float
    clipped_fraction: float
    downscaled: bool
    processing_scale: float
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "megapixels": round(self.megapixels, 3),
            "blur_score": round(self.blur_score, 2),
            "mean_intensity": round(self.mean_intensity, 2),
            "intensity_std": round(self.intensity_std, 2),
            "clipped_fraction": round(self.clipped_fraction, 4),
            "downscaled": self.downscaled,
            "processing_scale": round(self.processing_scale, 6),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PreprocessedImage:
    """Structured result of the preprocessing step.

    Attributes:
        normalized: finite ``uint8`` grayscale image (input normalized).
        enhanced: CLAHE-enhanced ``uint8`` grayscale image, same shape.
        source_shape: shape of the input accepted by ``process``.
        quality: advisory image measurements and preflight warnings.
    """

    normalized: np.ndarray
    enhanced: np.ndarray
    source_shape: tuple
    quality: ImageQuality | None = None

    @property
    def shape(self) -> tuple:
        """Shape of the normalized/enhanced grayscale output."""
        return self.normalized.shape

    @property
    def processing_scale(self) -> float:
        """Ratio of processed width to accepted input width."""
        if self.quality is not None:
            return self.quality.processing_scale
        width = int(self.source_shape[1]) if len(self.source_shape) >= 2 else 0
        return self.enhanced.shape[1] / width if width else 1.0

    @property
    def original_width(self) -> int:
        return int(self.source_shape[1]) if len(self.source_shape) >= 2 else int(self.enhanced.shape[1])

    @property
    def original_height(self) -> int:
        return int(self.source_shape[0]) if len(self.source_shape) >= 1 else int(self.enhanced.shape[0])


class ImagePreprocessor:
    """Normalizes and enhances arbitrary grayscale or colour images.

    ``process`` accepts either an image path or a raw ``np.ndarray`` and
    returns a :class:`PreprocessedImage`.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_side: int = DEFAULT_MIN_SIDE,
    ) -> None:
        self.enhancer = ClaheEnhancer(
            ClaheConfig(clip_limit=clip_limit, tile_grid_size=tuple(tile_grid_size))
        )
        self.max_pixels = max(0, int(max_pixels))
        self.min_side = max(1, int(min_side))

    @staticmethod
    def load(image_path: PathLike) -> np.ndarray:
        """Load a grayscale image from disk with decode validation.

        OpenCV honours EXIF orientation by default for JPEG input. Invalid or
        unreadable files raise ``ValueError`` with a client-safe message.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Could not load image at {image_path}")
        buffer = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image could not be decoded: {path.name}")
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

    def _validate(self, array: np.ndarray) -> None:
        if array.ndim not in (2, 3):
            raise ValueError("Image must be a 2D grayscale or 3D colour array")
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            raise ValueError("Image must have 1, 3, or 4 channels")
        if array.size == 0 or min(array.shape[0], array.shape[1]) < self.min_side:
            raise ValueError(
                f"Image is too small: minimum side is {self.min_side}px"
            )
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            raise ValueError("Image array must be numeric")

    def _bounded(self, array: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = array.shape[:2]
        pixels = int(height) * int(width)
        if not self.max_pixels or pixels <= self.max_pixels:
            return array, 1.0
        scale = (self.max_pixels / float(pixels)) ** 0.5
        target_w = max(1, int(round(width * scale)))
        target_h = max(1, int(round(height * scale)))
        resized = cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized, target_w / float(width)

    @staticmethod
    def _quality(normalized: np.ndarray, scale: float) -> ImageQuality:
        height, width = normalized.shape[:2]
        blur = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
        mean = float(np.mean(normalized))
        std = float(np.std(normalized))
        clipped = float(np.mean((normalized <= 2) | (normalized >= 253)))
        warnings = []
        if min(width, height) < 256:
            warnings.append("LOW_RESOLUTION")
        if blur < 12.0:
            warnings.append("LOW_SHARPNESS")
        if std < 12.0:
            warnings.append("LOW_CONTRAST")
        if clipped > 0.35:
            warnings.append("EXCESSIVE_CLIPPING")
        if scale < 0.999:
            warnings.append("DOWNSAMPLED_FOR_ANALYSIS")
        return ImageQuality(
            width=width,
            height=height,
            megapixels=(width * height) / 1_000_000.0,
            blur_score=blur,
            mean_intensity=mean,
            intensity_std=std,
            clipped_fraction=clipped,
            downscaled=scale < 0.999,
            processing_scale=scale,
            warnings=tuple(warnings),
        )

    def process(self, image: PathLike | ImageArray) -> PreprocessedImage:
        """Validate, optionally downscale, normalize, and enhance an image."""
        if isinstance(image, (str, Path)):
            image = self.load(image)
        array = np.asarray(image)
        self._validate(array)
        bounded, scale = self._bounded(array)
        normalized = normalize(bounded)
        enhanced = self.enhancer.enhance(normalized)
        quality = self._quality(normalized, scale)
        return PreprocessedImage(
            normalized=normalized,
            enhanced=enhanced,
            source_shape=array.shape,
            quality=quality,
        )


__all__ = [
    "ClaheConfig",
    "ClaheEnhancer",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MIN_SIDE",
    "ImageArray",
    "ImagePreprocessor",
    "ImageQuality",
    "PreprocessedImage",
    "enhance",
    "normalize",
    "normalize_to_uint8",
    "to_grayscale",
]
