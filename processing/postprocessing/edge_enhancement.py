"""Edge enhancement for the Finish stage."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def enhance_edges(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Add a controlled Laplacian edge signal to the source image."""

    validate_uint8_image(image)
    _validate_strength(strength)
    color_image, alpha = split_alpha(image)
    edges = cv2.Laplacian(color_image, cv2.CV_64F, ksize=3)
    enhanced = np.clip(
        np.rint(color_image.astype(np.float64) + strength * edges), 0, 255
    ).astype(np.uint8)
    return restore_alpha(enhanced, alpha)


def _validate_strength(strength: float) -> None:
    if (
        isinstance(strength, bool)
        or not isinstance(strength, Real)
        or not np.isfinite(strength)
        or strength < 0
    ):
        raise ValueError("strength must be a finite number greater than or equal to zero.")
