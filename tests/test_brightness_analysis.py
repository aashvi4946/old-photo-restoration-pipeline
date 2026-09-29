import numpy as np
import pytest

from processing.analysis.brightness import calculate_brightness


def test_black_image_brightness():
    image = np.zeros((100, 100), dtype=np.uint8)

    assert calculate_brightness(image) == 0.0


def test_white_image_brightness():
    image = np.full((100, 100), 255, dtype=np.uint8)

    assert calculate_brightness(image) == 255.0


def test_mid_gray_brightness():
    image = np.full((100, 100), 128, dtype=np.uint8)

    assert calculate_brightness(image) == 128.0


def test_color_image_is_supported():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    score = calculate_brightness(image)

    assert isinstance(score, float)


def test_bgra_image_is_supported():
    image = np.full((100, 100, 4), (128, 128, 128, 20), dtype=np.uint8)

    assert calculate_brightness(image) == pytest.approx(128.0)


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        calculate_brightness(image)
