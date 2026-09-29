from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a NumPy array."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None or image.size == 0:
        raise ValueError(f"Unable to decode image: {path}")

    if image.ndim not in (2, 3):
        raise ValueError("Unsupported image dimensions.")

    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Unsupported channel count.")

    return image
