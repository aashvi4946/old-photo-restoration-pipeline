import cv2
import numpy as np


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using variance of the Laplacian."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Unsupported image dimensions.")

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_blur(score: float) -> str:
    """Classify blur using initial configurable thresholds."""

    if score < 100:
        return "High"
    elif score < 300:
        return "Medium"
    return "Low"
