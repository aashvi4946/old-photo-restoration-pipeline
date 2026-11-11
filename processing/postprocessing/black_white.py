"""Black-and-white conversion for the Finish stage."""

from __future__ import annotations

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def convert_to_black_and_white(image: np.ndarray) -> np.ndarray:
    """Remove color while retaining BGR/BGRA shape and alpha where present."""

    validate_uint8_image(image)
    color_image, alpha = split_alpha(image)
    if color_image.ndim == 2:
        return color_image.copy()

    grayscale = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    monochrome = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    return restore_alpha(monochrome, alpha)
