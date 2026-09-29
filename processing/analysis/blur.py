import cv2
import numpy as np


DEFAULT_BLUR_MEDIUM_THRESHOLD = 100.0
DEFAULT_BLUR_LOW_THRESHOLD = 300.0


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using variance of the Laplacian."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported channel count.")
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Unsupported image dimensions.")

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_blur(
    score: float,
    medium_threshold: float = DEFAULT_BLUR_MEDIUM_THRESHOLD,
    low_threshold: float = DEFAULT_BLUR_LOW_THRESHOLD,
) -> str:
    """Classify blur using configurable thresholds."""

    if medium_threshold >= low_threshold:
        raise ValueError("medium_threshold must be less than low_threshold.")

    if score < medium_threshold:
        return "High"
    elif score < low_threshold:
        return "Medium"
    return "Low"
