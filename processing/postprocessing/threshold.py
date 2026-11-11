"""Adaptive thresholding exposed through the Advanced pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def apply_adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    constant: float = 2.0,
    method: str = "gaussian",
) -> np.ndarray:
    """Apply adaptive thresholding while retaining BGR/BGRA image structure."""

    validate_uint8_image(image)
    _validate_parameters(block_size, constant, method)
    color_image, alpha = split_alpha(image)
    gray = (
        color_image
        if color_image.ndim == 2
        else cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    )
    threshold_method = {
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    }[method]
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        threshold_method,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )
    if color_image.ndim == 2:
        return thresholded
    return restore_alpha(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR), alpha)


def _validate_parameters(block_size: int, constant: float, method: str) -> None:
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 1
        or block_size % 2 == 0
    ):
        raise ValueError("block_size must be an odd integer greater than one.")
    if isinstance(constant, bool) or not isinstance(constant, (int, float)) or not np.isfinite(constant):
        raise ValueError("constant must be a finite number.")
    if method not in {"mean", "gaussian"}:
        raise ValueError("method must be 'mean' or 'gaussian'.")
