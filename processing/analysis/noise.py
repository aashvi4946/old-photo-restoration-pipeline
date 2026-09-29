import cv2
import numpy as np


def calculate_noise_score(image: np.ndarray) -> float:
    """Estimate image noise using a robust MAD-based method."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Unsupported image dimensions.")

    gray = gray.astype(np.float32)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    median = np.median(laplacian)
    mad = np.median(np.abs(laplacian - median))

    return float(mad / 0.6745)


def classify_noise(score: float) -> str:
    """Classify estimated noise level using initial thresholds."""

    if score < 5:
        return "Low"
    elif score < 15:
        return "Medium"
    return "High"
