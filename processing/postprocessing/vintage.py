"""Vintage styling for the Finish stage."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def apply_vintage_style(image: np.ndarray, intensity: float = 0.35) -> np.ndarray:
    """Blend in a restrained sepia tone and softly reduce contrast."""

    validate_uint8_image(image)
    _validate_intensity(intensity)
    color_image, alpha = split_alpha(image)
    if color_image.ndim == 2:
        softened = (color_image.astype(np.float64) - 127.5) * (1 - 0.15 * intensity) + 127.5
        return np.clip(np.rint(softened), 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float64)
    sepia_matrix = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        dtype=np.float64,
    )
    sepia_rgb = np.clip(rgb @ sepia_matrix.T, 0, 255)
    blended = rgb * (1 - intensity) + sepia_rgb * intensity
    softened = (blended - 127.5) * (1 - 0.15 * intensity) + 127.5
    styled = np.clip(np.rint(softened), 0, 255).astype(np.uint8)
    return restore_alpha(cv2.cvtColor(styled, cv2.COLOR_RGB2BGR), alpha)


def _validate_intensity(intensity: float) -> None:
    if (
        isinstance(intensity, bool)
        or not isinstance(intensity, Real)
        or not np.isfinite(intensity)
        or not 0 <= intensity <= 1
    ):
        raise ValueError("intensity must be a finite number from 0 to 1.")
