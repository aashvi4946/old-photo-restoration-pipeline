import numpy as np

from models.image_profile import ImageProfile
from processing.analysis.metadata import analyze_metadata
from processing.analysis.blur import calculate_blur_score, classify_blur
from processing.analysis.noise import calculate_noise_score, classify_noise
from processing.analysis.brightness import calculate_brightness
from processing.analysis.contrast import calculate_contrast
from processing.analysis.histogram import calculate_histogram
from processing.analysis.color_cast import detect_color_cast


def analyze_image(image: np.ndarray) -> ImageProfile:
    """Run the complete image analysis pipeline."""

    if image is None or image.size == 0:
        raise ValueError("Image cannot be empty.")

    profile = analyze_metadata(image)

    blur_score = calculate_blur_score(image)
    noise_score = calculate_noise_score(image)

    profile.blur_score = blur_score
    profile.blur_level = classify_blur(blur_score)

    profile.noise_score = noise_score
    profile.noise_level = classify_noise(noise_score)

    profile.brightness = calculate_brightness(image)
    profile.contrast = calculate_contrast(image)
    profile.histogram = calculate_histogram(image)
    profile.color_cast = detect_color_cast(image)

    return profile
