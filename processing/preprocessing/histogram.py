"""Global histogram-equalization preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from processing.preprocessing._image_utils import (
    restore_alpha,
    split_alpha,
    validate_uint8_image,
)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Improve global contrast without changing image dimensions or alpha.

    Grayscale images are equalized directly. BGR and BGRA images are converted
    to YCrCb so only their luminance channel is equalized, which avoids the
    unnatural color shifts caused by independently equalizing color channels.
    """

    validate_uint8_image(image)

    color_image, alpha = split_alpha(image)

    if color_image.ndim == 2:
        return cv2.equalizeHist(color_image)

    ycrcb = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return restore_alpha(equalized, alpha)
