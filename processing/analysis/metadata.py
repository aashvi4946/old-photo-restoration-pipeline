import numpy as np

from models.image_profile import ImageProfile


def analyze_metadata(image: np.ndarray) -> ImageProfile:
    """Extract basic metadata from an image."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        is_grayscale = True

    elif image.ndim == 3:
        height, width, channels = image.shape

        if channels not in (3, 4):
            raise ValueError(f"Unsupported channel count: {channels}")

        is_grayscale = False

    else:
        raise ValueError("Unsupported image dimensions.")

    return ImageProfile(
        width=width,
        height=height,
        channels=channels,
        is_grayscale=is_grayscale,
    )
