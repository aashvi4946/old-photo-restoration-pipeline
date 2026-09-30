"""Shared image validation and alpha helpers for processing operations."""

from __future__ import annotations

import numpy as np


def validate_uint8_image(image: np.ndarray) -> None:
    """Validate the project's supported grayscale, BGR, or BGRA image format."""

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.dtype != np.uint8:
        raise TypeError("Image dtype must be uint8.")

    if image.ndim == 2:
        return

    if image.ndim == 3 and image.shape[2] in (3, 4):
        return

    raise ValueError("Image must be grayscale, BGR, or BGRA.")


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return color data and an optional alpha channel without modifying input."""

    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]

    return image, None


def restore_alpha(
    image: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray:
    """Attach an unchanged alpha channel to a processed BGR image when present."""

    if alpha is None:
        return image

    return np.dstack((image, alpha))
