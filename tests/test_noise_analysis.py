import numpy as np
import pytest

from processing.analysis.noise import calculate_noise_score, classify_noise


def test_noise_score_returns_float():
    image = np.zeros((100, 100), dtype=np.uint8)

    score = calculate_noise_score(image)

    assert isinstance(score, float)
    assert score >= 0


def test_color_image_is_supported():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    score = calculate_noise_score(image)

    assert isinstance(score, float)


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        calculate_noise_score(image)


def test_noise_classification():
    assert classify_noise(2) == "Low"
    assert classify_noise(10) == "Medium"
    assert classify_noise(20) == "High"
