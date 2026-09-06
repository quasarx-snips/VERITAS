"""VERITAS preprocessing tests (P0.1 migration of the reference preprocessor).

The preprocessing module must work for arbitrary grayscale or colour image
pairs (no lunar/terrain assumptions); all tests use synthetic arrays.
"""

import pathlib

import cv2
import numpy as np
import pytest

from veritas.preprocessing import (
    ClaheConfig,
    ClaheEnhancer,
    ImagePreprocessor,
    PreprocessedImage,
    normalize,
    to_grayscale,
)


def _gray_image(shape=(48, 64), dtype=np.uint8, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape).astype(dtype)


# ---------------------------------------------------------------------------
# 1. grayscale input
# ---------------------------------------------------------------------------
def test_grayscale_input_normalizes_to_same_shape_uint8():
    gray = _gray_image((40, 40))
    out = ImagePreprocessor().normalize(gray)
    assert out.dtype == np.uint8
    assert out.shape == (40, 40)


def test_grayscale_input_matches_reference_min_max_rescale():
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    normalized = normalize(image)
    assert normalized.dtype == np.uint8
    assert int(normalized.min()) == 0
    assert int(normalized.max()) == 255


# ---------------------------------------------------------------------------
# 2. BGR input
# ---------------------------------------------------------------------------
def test_bgr_input_reduces_to_single_channel():
    bgr = np.zeros((20, 30, 3), np.uint8)
    bgr[..., 2] = 200
    normalized = ImagePreprocessor().normalize(bgr)
    assert normalized.dtype == np.uint8
    assert normalized.ndim == 2
    assert normalized.shape == (20, 30)
    assert to_grayscale(bgr).ndim == 2


# ---------------------------------------------------------------------------
# 3. BGRA input
# ---------------------------------------------------------------------------
def test_bgra_input_reduces_to_single_channel():
    bgra = np.zeros((20, 30, 4), np.uint8)
    bgra[..., 1] = 120
    normalized = ImagePreprocessor().normalize(bgra)
    assert normalized.dtype == np.uint8
    assert normalized.ndim == 2
    assert normalized.shape == (20, 30)


def test_colour_with_unsupported_channel_count_raises():
    with pytest.raises(ValueError):
        ImagePreprocessor().normalize(np.zeros((5, 5, 2), np.uint8))


# ---------------------------------------------------------------------------
# 4. uint8 input
# ---------------------------------------------------------------------------
def test_uint8_input_is_copied_not_mutated():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    original = image.copy()
    out = ImagePreprocessor().normalize(image)
    out[0, 0] = 0
    assert np.array_equal(image, original), "input must not be mutated"


# ---------------------------------------------------------------------------
# 5. float input
# ---------------------------------------------------------------------------
def test_float_input():
    image = np.linspace(0.0, 255.0, 60 * 60, dtype=np.float32).reshape(60, 60)
    out = ImagePreprocessor().normalize(image)
    assert out.dtype == np.uint8
    assert out.shape == (60, 60)
    assert int(out.min()) == 0
    assert int(out.max()) == 255


# ---------------------------------------------------------------------------
# 6. NaN/Inf handling
# ---------------------------------------------------------------------------
def test_nan_and_inf_handling():
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 255, (30, 40)).astype(np.float32)
    image[0, 0] = np.nan
    image[1, 1] = np.inf
    image[2, 2] = -np.inf
    out = ImagePreprocessor().normalize(image)
    assert np.isfinite(out).all()
    assert out.dtype == np.uint8
    assert out.shape == (30, 40)


def test_all_nonfinite_image_raises():
    with pytest.raises(ValueError):
        ImagePreprocessor().normalize(np.full((5, 5), np.nan, np.float32))


# ---------------------------------------------------------------------------
# 7. constant image
# ---------------------------------------------------------------------------
def test_constant_image_returns_zeros():
    constant = np.full((10, 12), 7.0, np.float32)
    out = ImagePreprocessor().normalize(constant)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 0


# ---------------------------------------------------------------------------
# 8. CLAHE execution
# ---------------------------------------------------------------------------
def test_clahe_execution():
    gray = _gray_image((80, 80))
    enhanced = ImagePreprocessor().enhance(gray)
    assert enhanced.dtype == np.uint8
    assert enhanced.shape == gray.shape
    enhancer = ClaheEnhancer(ClaheConfig(clip_limit=2.5, tile_grid_size=(8, 8)))
    assert np.array_equal(enhancer.enhance(gray), enhanced)


def test_process_runs_clahe_and_returns_structured_result():
    gray = _gray_image((90, 120))
    result = ImagePreprocessor().process(gray)
    assert isinstance(result, PreprocessedImage)
    assert result.normalized.dtype == np.uint8
    assert result.enhanced.dtype == np.uint8
    assert result.normalized.shape == result.enhanced.shape == (90, 120)


# ---------------------------------------------------------------------------
# 9. output dimensions preserved
# ---------------------------------------------------------------------------
def test_output_dimensions_preserved_for_all_input_kinds():
    pre = ImagePreprocessor()
    for shape in ((40, 50), (40, 50, 3), (40, 50, 4)):
        image = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
        result = pre.process(image)
        assert result.normalized.ndim == 2
        assert result.enhanced.ndim == 2
        assert result.normalized.shape == result.enhanced.shape
        assert result.normalized.shape[:2] == shape[:2]
        assert result.source_shape == shape


# ---------------------------------------------------------------------------
# 10. deterministic output for identical input/config
# ---------------------------------------------------------------------------
def test_deterministic_output_for_identical_input_and_config():
    gray = _gray_image((64, 64), seed=42)
    first = ImagePreprocessor(clip_limit=2.5, tile_grid_size=(8, 8)).process(gray)
    second = ImagePreprocessor(clip_limit=2.5, tile_grid_size=(8, 8)).process(gray)
    assert np.array_equal(first.normalized, second.normalized)
    assert np.array_equal(first.enhanced, second.enhanced)


# ---------------------------------------------------------------------------
# path-based I/O (facade convenience)
# ---------------------------------------------------------------------------
def test_process_loads_image_from_disk(tmp_path: pathlib.Path):
    image = _gray_image((90, 120))
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    result = ImagePreprocessor().process(path)
    assert result.normalized.dtype == result.enhanced.dtype == np.uint8
    assert result.normalized.shape == result.enhanced.shape == (90, 120)


def test_process_missing_file_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        ImagePreprocessor().process(tmp_path / "missing.png")


def test_rejects_empty_and_invalid_arrays():
    with pytest.raises(ValueError):
        ImagePreprocessor().normalize(np.empty((0, 5)))
    with pytest.raises(ValueError):
        ImagePreprocessor().normalize(np.zeros(6, np.uint8))