"""Guided classical inpainting operations for photo repair."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


_METHODS = {"telea": cv2.INPAINT_TELEA, "navier_stokes": cv2.INPAINT_NS}


def apply_inpainting(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: float = 3.0,
) -> np.ndarray:
    """Repair selected pixels with Telea or Navier–Stokes inpainting."""

    validate_uint8_image(image)
    normalized_mask = _validate_and_normalize_mask(mask, image.shape[:2])
    method_flag = _validate_method(method)
    _validate_radius(radius)
    color_image, alpha = split_alpha(image)
    repaired = cv2.inpaint(color_image, normalized_mask, float(radius), method_flag)
    return restore_alpha(repaired, alpha)


def _validate_and_normalize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array.")
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2D array.")
    if mask.shape != image_shape:
        raise ValueError("mask dimensions must match the image dimensions.")
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError("mask must contain numeric values.")
    if not np.any(mask):
        raise ValueError("Select an area to repair before applying inpainting.")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validate_method(method: str) -> int:
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError("method must be 'telea' or 'navier_stokes'.")
    return _METHODS[method]


def _validate_radius(radius: float) -> None:
    if (
        isinstance(radius, bool)
        or not isinstance(radius, Real)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("radius must be a finite number greater than zero.")
