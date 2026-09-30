"""Explicit image-resizing preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from processing._image_utils import validate_uint8_image


_VALID_INTERPOLATIONS = frozenset(range(cv2.INTER_MAX))


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize an image explicitly to ``width`` × ``height`` pixels.

    All supported image channels are resized together, so BGR/BGRA channel
    structure is preserved. ``INTER_AREA`` is the default because it is a
    suitable choice for downscaling; the caller may explicitly select another
    validated OpenCV interpolation method.
    """

    validate_uint8_image(image)
    _validate_parameters(width, height, interpolation)

    return cv2.resize(
        image,
        (width, height),
        interpolation=interpolation,
    )


def _validate_parameters(width: int, height: int, interpolation: int) -> None:
    """Validate destination dimensions and interpolation method."""

    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if (
        isinstance(interpolation, bool)
        or not isinstance(interpolation, int)
        or interpolation not in _VALID_INTERPOLATIONS
    ):
        raise ValueError("interpolation must be a supported OpenCV method.")
