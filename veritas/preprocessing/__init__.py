"""Image loading, normalisation, and contrast enhancement primitives.

P0.1: migrated — the proven reference ``ImagePreprocessor`` implementation,
adapted as ``Preprocessor`` (grayscale/BGR/BGRA/float -> finite uint8, CLAHE).
"""

from .preprocessor import ImageArray, Preprocessor

__all__ = ["ImageArray", "Preprocessor"]