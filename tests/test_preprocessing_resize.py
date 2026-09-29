import cv2
import numpy as np
import pytest

from processing.preprocessing.resize import resize_image


def test_resize_image_changes_grayscale_dimensions():
    image = np.zeros((100, 100), dtype=np.uint8)

    result = resize_image(image, width=50, height=50)

    assert result.shape == (50, 50)
    assert result.dtype == np.uint8


def test_resize_image_changes_bgr_dimensions_without_channel_loss():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = resize_image(image, width=80, height=40)

    assert result.shape == (40, 80, 3)
    assert result.dtype == np.uint8


def test_resize_image_preserves_bgra_channel_structure_and_alpha_data():
    image = np.full((100, 100, 4), (20, 40, 60, 123), dtype=np.uint8)

    result = resize_image(image, width=50, height=50)

    assert result.shape == (50, 50, 4)
    assert result.dtype == np.uint8
    assert np.all(result[:, :, 3] == 123)


def test_resize_image_accepts_explicit_interpolation():
    image = np.zeros((100, 100), dtype=np.uint8)

    result = resize_image(
        image,
        width=50,
        height=50,
        interpolation=cv2.INTER_LINEAR,
    )

    assert result.shape == (50, 50)


def test_resize_image_does_not_mutate_input():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    original = image.copy()

    resize_image(image, width=5, height=5)

    assert np.array_equal(image, original)


@pytest.mark.parametrize("width", [0, -1, 5.5, True])
def test_resize_image_rejects_invalid_width(width):
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_image(image, width=width, height=5)


@pytest.mark.parametrize("height", [0, -1, 5.5, True])
def test_resize_image_rejects_invalid_height(height):
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_image(image, width=5, height=height)


@pytest.mark.parametrize("interpolation", [-1, cv2.INTER_MAX, "linear", True])
def test_resize_image_rejects_invalid_interpolation(interpolation):
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_image(image, width=5, height=5, interpolation=interpolation)
