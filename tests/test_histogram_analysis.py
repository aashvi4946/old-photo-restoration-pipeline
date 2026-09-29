import numpy as np
import pytest

from processing.analysis.histogram import calculate_histogram


def test_histogram_has_256_bins():
    image = np.zeros((100, 100), dtype=np.uint8)

    histogram = calculate_histogram(image)

    assert histogram.shape == (256,)


def test_histogram_pixel_count():
    image = np.zeros((100, 100), dtype=np.uint8)

    histogram = calculate_histogram(image)

    assert histogram.sum() == image.size


def test_black_image_histogram():
    image = np.zeros((100, 100), dtype=np.uint8)

    histogram = calculate_histogram(image)

    assert histogram[0] == 10000
    assert histogram[255] == 0


def test_white_image_histogram():
    image = np.full((100, 100), 255, dtype=np.uint8)

    histogram = calculate_histogram(image)

    assert histogram[255] == 10000
    assert histogram[0] == 0


def test_color_image_is_supported():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    histogram = calculate_histogram(image)

    assert histogram.shape == (256,)


def test_bgra_image_is_supported():
    image = np.full((100, 100, 4), (128, 128, 128, 20), dtype=np.uint8)

    histogram = calculate_histogram(image)

    assert histogram.shape == (256,)


def test_empty_image_raises_error():
    image = np.array([])

    with pytest.raises(ValueError):
        calculate_histogram(image)
