import numpy as np
import pytest

from processing.analysis.metadata import analyze_metadata


def test_color_image_metadata():
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    profile = analyze_metadata(image)

    assert profile.width == 800
    assert profile.height == 600
    assert profile.channels == 3
    assert profile.is_grayscale is False


def test_grayscale_image_metadata():
    image = np.zeros((600, 800), dtype=np.uint8)

    profile = analyze_metadata(image)

    assert profile.width == 800
    assert profile.height == 600
    assert profile.channels == 1
    assert profile.is_grayscale is True


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        analyze_metadata(image)


def test_invalid_dimensions_raise_error():
    image = np.zeros((10, 10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        analyze_metadata(image)
