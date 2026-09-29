import cv2
import numpy as np


DEFAULT_NOISE_MEDIUM_THRESHOLD = 5.0
DEFAULT_NOISE_HIGH_THRESHOLD = 15.0


def calculate_noise_score(image: np.ndarray) -> float:
    """Estimate image noise using a robust MAD-based method."""

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

    gray = gray.astype(np.float32)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    median = np.median(laplacian)
    mad = np.median(np.abs(laplacian - median))

    return float(mad / 0.6745)


def classify_noise(
    score: float,
    medium_threshold: float = DEFAULT_NOISE_MEDIUM_THRESHOLD,
    high_threshold: float = DEFAULT_NOISE_HIGH_THRESHOLD,
) -> str:
    """Classify estimated noise level using configurable thresholds."""

    if medium_threshold >= high_threshold:
        raise ValueError("medium_threshold must be less than high_threshold.")

    if score < medium_threshold:
        return "Low"
    elif score < high_threshold:
        return "Medium"
    return "High"
