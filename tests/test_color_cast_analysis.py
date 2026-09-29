import numpy as np
import pytest

from processing.analysis.color_cast import detect_color_cast


def test_no_color_cast():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    assert detect_color_cast(image) == "None"


def test_red_color_cast():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 2] = 200

    assert detect_color_cast(image) == "Red"


def test_green_color_cast():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 1] = 200

    assert detect_color_cast(image) == "Green"


def test_blue_color_cast():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 200

    assert detect_color_cast(image) == "Blue"


def test_grayscale_image():
    image = np.full((100, 100), 128, dtype=np.uint8)

    assert detect_color_cast(image) == "None"


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        detect_color_cast(image)
