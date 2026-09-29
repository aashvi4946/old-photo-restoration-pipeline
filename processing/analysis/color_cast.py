import cv2
import numpy as np


def detect_color_cast(image: np.ndarray) -> str:
    """Detect the dominant color cast in a BGR image."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 2:
        return "None"

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Unsupported image dimensions.")

    means = np.mean(image, axis=(0, 1))
    blue, green, red = means

    max_mean = max(blue, green, red)
    min_mean = min(blue, green, red)

    # Ignore very small channel differences.
    if max_mean - min_mean < 10:
        return "None"

    if red == max_mean:
        return "Red"

    if green == max_mean:
        return "Green"

    return "Blue"
