"""Contrast Limited Adaptive Histogram Equalization preprocessing."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing.preprocessing._image_utils import (
    restore_alpha,
    split_alpha,
    validate_uint8_image,
)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply local, contrast-limited enhancement to image luminance.

    Grayscale images are processed directly. BGR and BGRA images are converted
    to YCrCb and processed only through their Y channel; BGRA alpha is retained
    unchanged.
    """

    validate_uint8_image(image)
    _validate_parameters(clip_limit, tile_grid_size)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    color_image, alpha = split_alpha(image)

    if color_image.ndim == 2:
        return clahe.apply(color_image)

    ycrcb = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return restore_alpha(enhanced, alpha)


def _validate_parameters(
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> None:
    """Validate OpenCV CLAHE parameters before constructing the operation."""

    if (
        isinstance(clip_limit, bool)
        or not isinstance(clip_limit, Real)
        or not np.isfinite(clip_limit)
        or clip_limit <= 0
    ):
        raise ValueError("clip_limit must be a finite number greater than zero.")

    if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2:
        raise ValueError("tile_grid_size must be a two-item tuple.")

    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in tile_grid_size
    ):
        raise ValueError("tile_grid_size dimensions must be positive integers.")
