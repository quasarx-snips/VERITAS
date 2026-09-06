"""VERITAS preprocessing — normalization.

Normalization responsibility: reduce arbitrary image arrays to a single finite
grayscale ``uint8`` channel.

Migrated from the reference repository's preprocessing module. The proven
numerical behavior is preserved:

- grayscale, BGR, and BGRA inputs are accepted;
- non-``uint8`` numeric dtypes are rescaled so the finite minimum maps to 0 and
  the finite maximum maps to 255;
- NaN values are replaced with the finite minimum; ``+Inf``/``-Inf`` are
  replaced with the finite maximum/minimum;
- a constant image normalizes to an all-zero ``uint8`` image;
- the input array is never mutated.

The module is domain-agnostic: it works for arbitrary grayscale or colour image
pairs and carries no lunar/terrain assumptions.

Dependencies: opencv-python, numpy.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

ImageArray = np.ndarray


def validate_image(image: Any) -> np.ndarray:
    """Return the input as an ndarray after basic emptiness/rank checks."""
    array = np.asarray(image)
    if array.size == 0 or array.ndim not in (2, 3):
        raise ValueError("image must be a non-empty grayscale or BGR/BGRA array")
    return array


def to_grayscale(image: Any) -> np.ndarray:
    """Reduce a BGR/BGRA array to a single grayscale channel.

    Grayscale (2-D) arrays are returned as a copy of the input.
    """
    array = validate_image(image)
    if array.ndim == 3:
        if array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("colour images must have 3 or 4 channels")
    return array


def normalize_to_uint8(image: Any) -> np.ndarray:
    """Return a finite grayscale uint8 image without mutating the input.

    The transformation is a linear rescale onto the full uint8 range:
    ``out = clip((values - lo) * (255 / (hi - lo)), 0, 255)`` where ``lo``/``hi``
    are the finite minimum/maximum intensities of the grayscale input.
    """
    array = validate_image(image)
    array = to_grayscale(array)
    if array.dtype == np.uint8:
        return array.copy()
    values = array.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("image contains no finite intensity values")
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo)
    return np.clip((values - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def normalize(image: Any) -> np.ndarray:
    """Normalize an arbitrary image array to a finite grayscale uint8 image.

    Convenience alias preserving the reference API name.
    """
    return normalize_to_uint8(image)