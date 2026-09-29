"""Classical automatic white-balance preprocessing."""

from __future__ import annotations

import numpy as np

from processing.preprocessing._image_utils import (
    restore_alpha,
    split_alpha,
    validate_uint8_image,
)


def auto_white_balance(image: np.ndarray) -> np.ndarray:
    """Apply gray-world white balance while preserving image representation.

    The gray-world assumption scales each BGR channel toward their shared mean.
    Grayscale images contain no color cast to correct and return an independent
    unchanged copy. BGRA alpha is retained unchanged.
    """

    validate_uint8_image(image)

    color_image, alpha = split_alpha(image)

    if color_image.ndim == 2:
        return color_image.copy()

    channel_means = color_image.mean(axis=(0, 1), dtype=np.float64)
    target_mean = float(channel_means.mean())
    scales = np.divide(
        target_mean,
        channel_means,
        out=np.ones_like(channel_means),
        where=channel_means > 0,
    )

    balanced = np.clip(
        np.rint(color_image.astype(np.float64) * scales),
        0,
        255,
    ).astype(np.uint8)

    return restore_alpha(balanced, alpha)
