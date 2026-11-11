"""Contrast adjustment for the Finish stage."""

from __future__ import annotations

from numbers import Real

import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Scale contrast around middle gray without modifying the input image."""

    validate_uint8_image(image)
    _validate_factor(factor)
    color_image, alpha = split_alpha(image)
    adjusted = np.clip(
        np.rint((color_image.astype(np.float64) - 127.5) * factor + 127.5),
        0,
        255,
    ).astype(np.uint8)
    return restore_alpha(adjusted, alpha)


def _validate_factor(factor: float) -> None:
    if (
        isinstance(factor, bool)
        or not isinstance(factor, Real)
        or not np.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError("factor must be a finite number greater than zero.")
