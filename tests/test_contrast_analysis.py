import numpy as np
import pytest

from processing.analysis.contrast import calculate_contrast


def test_constant_black_image():
    image = np.zeros((100, 100), dtype=np.uint8)

    assert calculate_contrast(image) == 0.0


def test_constant_white_image():
    image = np.full((100, 100), 255, dtype=np.uint8)

    assert calculate_contrast(image) == 0.0


def test_mixed_image_has_contrast():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 50:] = 255

    assert calculate_contrast(image) > 0.0


def test_color_image_is_supported():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    score = calculate_contrast(image)

    assert isinstance(score, float)


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        calculate_contrast(image)
