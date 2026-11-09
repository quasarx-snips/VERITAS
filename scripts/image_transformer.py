"""Synthetic image-pair transformation utilities for generating test datasets."""

from __future__ import annotations

import os
import random
from typing import Any

import cv2
import numpy as np

#: Indices of raw images to transform into the processed set.
RAW_IMAGE_START_INDEX: int = 51
RAW_IMAGE_END_INDEX: int = 100

#: Rotation angle range in degrees.
ROTATION_ANGLE_MIN: float = -30.0
ROTATION_ANGLE_MAX: float = 30.0

#: Blur kernel size options.
BLUR_KERNEL_SIZES: tuple[int, ...] = (3, 5, 7)

#: Brightness adjustment range.
BRIGHTNESS_MIN: int = -50
BRIGHTNESS_MAX: int = 50

#: Contrast adjustment range.
CONTRAST_MIN: float = 0.5
CONTRAST_MAX: float = 1.5

#: Stretch/zoom scale range.
STRETCH_SCALE_MIN: float = 0.8
STRETCH_SCALE_MAX: float = 1.2

#: Available transformation names.
TRANSFORMATIONS: tuple[str, ...] = (
    "rotate",
    "blur",
    "brightness_contrast",
    "stretch_zoom",
)


def rotate_image(image: Any, angle: float) -> Any:
    """Rotate an image around its center by ``angle`` degrees."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def blur_image(image: Any, kernel_size: int) -> Any:
    """Apply Gaussian blur with a square kernel of ``kernel_size``."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def change_brightness_contrast(image: Any, brightness: int, contrast: float) -> Any:
    """Adjust brightness and contrast via ``convertScaleAbs``."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def stretch_zoom_image(image: Any, fx: float, fy: float) -> Any:
    """Resize an image by scale factors ``fx`` and ``fy``."""
    return cv2.resize(image, None, fx=fx, fy=fy)


def process_images(input_dir: str, output_dir: str) -> None:
    """Generate transformed copies of ``ds_51.jpg``..``ds_100.jpg`` into ``output_dir``."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(RAW_IMAGE_START_INDEX, RAW_IMAGE_END_INDEX + 1):
        filename = f"ds_{i}.jpg"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(input_path):
            image = cv2.imread(input_path)
            transformation = random.choice(TRANSFORMATIONS)

            if transformation == "rotate":
                angle = random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
                transformed_image = rotate_image(image, angle)
            elif transformation == "blur":
                kernel_size = random.choice(BLUR_KERNEL_SIZES)
                transformed_image = blur_image(image, kernel_size)
            elif transformation == "brightness_contrast":
                brightness = random.randint(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
                contrast = random.uniform(CONTRAST_MIN, CONTRAST_MAX)
                transformed_image = change_brightness_contrast(image, brightness, contrast)
            elif transformation == "stretch_zoom":
                fx = random.uniform(STRETCH_SCALE_MIN, STRETCH_SCALE_MAX)
                fy = random.uniform(STRETCH_SCALE_MIN, STRETCH_SCALE_MAX)
                transformed_image = stretch_zoom_image(image, fx, fy)
            else:
                continue

            cv2.imwrite(output_path, transformed_image)
            print(f"Processed {filename} with {transformation} and saved to {output_path}")


if __name__ == "__main__":
    input_directory = "data/raw"
    output_directory = "data/processed"
    process_images(input_directory, output_directory)