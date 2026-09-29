import cv2
import numpy as np


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate image contrast using standard deviation."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Unsupported image dimensions.")

    return float(np.std(gray))
