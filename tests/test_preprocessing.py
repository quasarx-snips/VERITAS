"""VERITAS preprocessing tests (P0.1 migration of the reference preprocessor)."""

import json
import pathlib

import cv2
import numpy as np
import pytest

from veritas.preprocessing import Preprocessor


def test_normalize_float_grayscale_to_uint8():
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    normalized = Preprocessor().normalize(image)
    assert normalized.dtype == np.uint8
    assert normalized.shape == (64, 64)
    assert int(normalized.min()) == 0
    assert int(normalized.max()) == 255


def test_normalize_float_colour_reduces_to_gray():
    image = np.dstack([np.linspace(0, 1, 64, dtype=np.float32)] * 3)
    normalized = Preprocessor().normalize(image)
    assert normalized.dtype == np.uint8
    assert normalized.shape == (1, 64)


def test_normalize_bgr_and_bgra_arrays():
    bgr = np.zeros((20, 30, 3), np.uint8)
    bgr[..., 2] = 200
    assert Preprocessor().normalize(bgr).shape == (20, 30)
    bgra = np.zeros((20, 30, 4), np.uint8)
    assert Preprocessor().normalize(bgra).shape == (20, 30)


def test_normalize_uint8_is_copied_not_mutated():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = Preprocessor().normalize(image)
    out[0, 0] = 0
    assert image[0, 0] == 0  # same value, but mutation must not corrupt input
    assert np.array_equal(image, np.arange(100, dtype=np.uint8).reshape(10, 10))


def test_normalize_constant_image_returns_zeros():
    constant = np.full((10, 12), 7.0, np.float32)
    out = Preprocessor().normalize(constant)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 0


def test_normalize_handles_nan_and_inf():
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 255, (30, 40)).astype(np.float32)
    image[0, 0] = np.nan
    image[1, 1] = np.inf
    image[2, 2] = -np.inf
    out = Preprocessor().normalize(image)
    assert np.isfinite(out).all()
    assert out.dtype == np.uint8
    assert out.shape == (30, 40)


def test_normalize_rejects_invalid_arrays():
    with pytest.raises(ValueError):
        Preprocessor().normalize(np.empty((0, 5)))
    with pytest.raises(ValueError):
        Preprocessor().normalize(np.zeros(6, np.uint8))
    with pytest.raises(ValueError):
        Preprocessor().normalize(np.zeros((5, 5, 2), np.uint8))


def test_enhance_preserves_shape_and_dtype():
    gray = np.linspace(0, 255, 64 * 80, dtype=np.uint8).reshape(64, 80)
    enhanced = Preprocessor().enhance(gray)
    assert enhanced.dtype == np.uint8
    assert enhanced.shape == gray.shape


def test_process_loads_and_preprocesses_from_disk(tmp_path: pathlib.Path):
    image = np.random.default_rng(3).integers(0, 256, (90, 120), dtype=np.uint8)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    raw, enhanced = Preprocessor().process(path)
    assert raw.dtype == enhanced.dtype == np.uint8
    assert raw.shape == enhanced.shape == (90, 120)


def test_process_missing_file_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        Preprocessor().process(tmp_path / "missing.png")