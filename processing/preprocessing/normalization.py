"""Explicit min-max intensity-normalization preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from processing._image_utils import (
    restore_alpha,
    split_alpha,
    validate_uint8_image,
)


def normalize_image(
    image: np.ndarray,
    new_min: int = 0,
    new_max: int = 255,
) -> np.ndarray:
    """Apply explicit min-max normalization to image intensity.

    Grayscale images are normalized directly. BGR and BGRA images have only
    luminance normalized in YCrCb space, preventing independent channel
    scaling from changing color balance. BGRA alpha is preserved unchanged.
    """

    validate_uint8_image(image)
    _validate_bounds(new_min, new_max)

    color_image, alpha = split_alpha(image)

    if color_image.ndim == 2:
        return _normalize_channel(color_image, new_min, new_max)

    ycrcb = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _normalize_channel(ycrcb[:, :, 0], new_min, new_max)
    normalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return restore_alpha(normalized, alpha)


def _validate_bounds(new_min: int, new_max: int) -> None:
    """Validate uint8 output bounds for min-max normalization."""

    if (
        isinstance(new_min, bool)
        or isinstance(new_max, bool)
        or not isinstance(new_min, int)
        or not isinstance(new_max, int)
        or not 0 <= new_min < new_max <= 255
    ):
        raise ValueError(
            "new_min and new_max must be integers where 0 <= new_min < "
            "new_max <= 255."
        )


def _normalize_channel(channel: np.ndarray, new_min: int, new_max: int) -> np.ndarray:
    """Normalize one uint8 intensity channel deterministically."""

    old_min = int(channel.min())
    old_max = int(channel.max())

    if old_min == old_max:
        return np.full_like(channel, new_min)

    normalized = (channel.astype(np.float64) - old_min) * (
        (new_max - new_min) / (old_max - old_min)
    ) + new_min

    return np.rint(normalized).astype(np.uint8)
