import numpy as np
import pytest

from processing.preprocessing.white_balance import auto_white_balance


def test_auto_white_balance_reduces_synthetic_color_cast():
    image = np.full((20, 20, 3), (50, 80, 120), dtype=np.uint8)

    result = auto_white_balance(image)

    input_channel_spread = np.ptp(image.mean(axis=(0, 1)))
    result_channel_spread = np.ptp(result.mean(axis=(0, 1)))

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result_channel_spread < input_channel_spread


def test_auto_white_balance_preserves_bgra_alpha_channel():
    image = np.full((20, 20, 4), (50, 80, 120, 0), dtype=np.uint8)
    image[:, 10:, 3] = 255

    result = auto_white_balance(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_auto_white_balance_returns_unchanged_grayscale_copy():
    image = np.array([[0, 50], [100, 255]], dtype=np.uint8)

    result = auto_white_balance(image)

    assert np.array_equal(result, image)
    assert result is not image


def test_auto_white_balance_handles_zero_value_color_channel():
    image = np.full((20, 20, 3), (0, 80, 120), dtype=np.uint8)

    result = auto_white_balance(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))
    assert np.all(result[:, :, 0] == 0)


def test_auto_white_balance_does_not_mutate_input():
    image = np.full((20, 20, 3), (50, 80, 120), dtype=np.uint8)
    original = image.copy()

    auto_white_balance(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "image, error_type",
    [
        (np.array([], dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 2), dtype=np.uint8), ValueError),
        (np.zeros((10, 10), dtype=np.float32), TypeError),
        (None, TypeError),
    ],
)
def test_auto_white_balance_rejects_invalid_images(image, error_type):
    with pytest.raises(error_type):
        auto_white_balance(image)
