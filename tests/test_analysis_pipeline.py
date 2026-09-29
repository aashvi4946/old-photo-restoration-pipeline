import numpy as np
import pytest

from processing.analysis.analyze import analyze_image


def test_analyze_image_returns_complete_profile():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 2] = 200

    profile = analyze_image(image)

    assert profile.width == 100
    assert profile.height == 100
    assert profile.channels == 3
    assert profile.is_grayscale is False

    assert isinstance(profile.blur_score, float)
    assert profile.blur_level in {"High", "Medium", "Low"}

    assert isinstance(profile.noise_score, float)
    assert profile.noise_level in {"Low", "Medium", "High"}

    assert isinstance(profile.brightness, float)
    assert isinstance(profile.contrast, float)

    assert isinstance(profile.histogram, np.ndarray)
    assert profile.histogram.shape == (256,)

    assert profile.color_cast == "Red"


def test_analyze_image_supports_grayscale():
    image = np.full((100, 100), 128, dtype=np.uint8)

    profile = analyze_image(image)

    assert profile.width == 100
    assert profile.height == 100
    assert profile.channels == 1
    assert profile.is_grayscale is True
    assert profile.color_cast == "None"


def test_analyze_image_rejects_empty_image():
    image = np.array([])

    with pytest.raises(ValueError):
        analyze_image(image)
