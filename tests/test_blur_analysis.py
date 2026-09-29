import numpy as np
import pytest

from processing.analysis.blur import calculate_blur_score, classify_blur


def test_blur_score_returns_float():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255

    score = calculate_blur_score(image)

    assert isinstance(score, float)
    assert score >= 0


def test_color_image_is_supported():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75] = 255

    score = calculate_blur_score(image)

    assert isinstance(score, float)


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        calculate_blur_score(image)


def test_blur_classification():
    assert classify_blur(50) == "High"
    assert classify_blur(150) == "Medium"
    assert classify_blur(500) == "Low"
