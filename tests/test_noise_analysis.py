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


def test_noise_classification_accepts_custom_thresholds():
    assert classify_noise(5, medium_threshold=10, high_threshold=20) == "Low"
    assert classify_noise(15, medium_threshold=10, high_threshold=20) == "Medium"
    assert classify_noise(25, medium_threshold=10, high_threshold=20) == "High"


def test_noise_classification_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        classify_noise(10, medium_threshold=20, high_threshold=10)


def test_four_channel_image_is_supported():
    image = np.zeros((100, 100, 4), dtype=np.uint8)

    score = calculate_noise_score(image)

    assert isinstance(score, float)


def test_invalid_noise_thresholds_raise_error():
    with pytest.raises(ValueError):
        classify_noise(10, medium_threshold=15, high_threshold=5)


def test_none_image_raises_error():
    with pytest.raises(ValueError):
        calculate_noise_score(None)
