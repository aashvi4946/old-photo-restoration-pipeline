"""Gamma correction for the Finish stage."""

from __future__ import annotations

from numbers import Real

import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction; values above one brighten midtones."""

    validate_uint8_image(image)
    _validate_gamma(gamma)
    table = np.clip(
        np.rint(255 * (np.arange(256, dtype=np.float64) / 255) ** (1 / gamma)),
        0,
        255,
    ).astype(np.uint8)
    color_image, alpha = split_alpha(image)
    return restore_alpha(table[color_image], alpha)


def _validate_gamma(gamma: float) -> None:
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, Real)
        or not np.isfinite(gamma)
        or gamma <= 0
    ):
        raise ValueError("gamma must be a finite number greater than zero.")
