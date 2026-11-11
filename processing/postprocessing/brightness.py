"""Brightness adjustment for the Finish stage."""

from __future__ import annotations

from numbers import Real

import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def adjust_brightness(image: np.ndarray, value: float = 0.0) -> np.ndarray:
    """Add a controlled brightness offset while retaining image structure."""

    validate_uint8_image(image)
    _validate_value(value)
    color_image, alpha = split_alpha(image)
    adjusted = np.clip(
        np.rint(color_image.astype(np.float64) + value), 0, 255
    ).astype(np.uint8)
    return restore_alpha(adjusted, alpha)


def _validate_value(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not -255 <= value <= 255
    ):
        raise ValueError("value must be a finite number from -255 to 255.")
