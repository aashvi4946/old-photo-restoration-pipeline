"""Controlled final sharpening for the Finish stage."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def sharpen_image(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Apply a controlled spatial sharpening kernel while retaining alpha."""

    validate_uint8_image(image)
    _validate_amount(amount)
    color_image, alpha = split_alpha(image)
    kernel = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    filtered = cv2.filter2D(color_image, cv2.CV_64F, kernel)
    sharpened = np.clip(
        np.rint(
            color_image.astype(np.float64) * (1 - amount) + filtered * amount
        ),
        0,
        255,
    ).astype(np.uint8)
    return restore_alpha(sharpened, alpha)


def _validate_amount(amount: float) -> None:
    if (
        isinstance(amount, bool)
        or not isinstance(amount, Real)
        or not np.isfinite(amount)
        or amount < 0
    ):
        raise ValueError("amount must be a finite number greater than or equal to zero.")
