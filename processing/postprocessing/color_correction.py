"""Manual color correction for the Finish stage."""

from __future__ import annotations

from numbers import Real

import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def apply_color_correction(
    image: np.ndarray,
    blue_gain: float = 1.0,
    green_gain: float = 1.0,
    red_gain: float = 1.0,
) -> np.ndarray:
    """Apply explicit BGR channel gains while preserving alpha unchanged."""

    validate_uint8_image(image)
    gains = (blue_gain, green_gain, red_gain)
    _validate_gains(gains)
    color_image, alpha = split_alpha(image)
    if color_image.ndim == 2:
        return color_image.copy()

    corrected = np.clip(
        np.rint(color_image.astype(np.float64) * np.asarray(gains)), 0, 255
    ).astype(np.uint8)
    return restore_alpha(corrected, alpha)


def _validate_gains(gains: tuple[float, float, float]) -> None:
    for name, gain in zip(("blue_gain", "green_gain", "red_gain"), gains):
        if (
            isinstance(gain, bool)
            or not isinstance(gain, Real)
            or not np.isfinite(gain)
            or gain <= 0
        ):
            raise ValueError(f"{name} must be a finite number greater than zero.")
