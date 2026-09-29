import numpy as np
import pytest

from processing.preprocessing.histogram import equalize_histogram


def test_equalize_histogram_supports_grayscale_images():
    image = np.array([[80, 90], [100, 110]], dtype=np.uint8)

    result = equalize_histogram(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result, np.array([[0, 85], [170, 255]], dtype=np.uint8))


def test_equalize_histogram_supports_bgr_images_without_channel_loss():
    image = np.array(
        [
            [[20, 40, 60], [40, 60, 80]],
            [[60, 80, 100], [80, 100, 120]],
        ],
        dtype=np.uint8,
    )

    result = equalize_histogram(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert np.all((result >= 0) & (result <= 255))


def test_equalize_histogram_preserves_bgra_alpha_channel():
    image = np.array(
        [
            [[20, 40, 60, 0], [40, 60, 80, 85]],
            [[60, 80, 100, 170], [80, 100, 120, 255]],
        ],
        dtype=np.uint8,
    )

    result = equalize_histogram(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_equalize_histogram_improves_low_contrast_grayscale_range():
    image = np.tile(np.arange(100, 111, dtype=np.uint8), (8, 1))

    result = equalize_histogram(image)

    assert int(result.max()) - int(result.min()) > int(image.max()) - int(image.min())


def test_equalize_histogram_does_not_mutate_input():
    image = np.array(
        [[[20, 40, 60, 10], [40, 60, 80, 20]]],
        dtype=np.uint8,
    )
    original = image.copy()

    equalize_histogram(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "image, error_type",
    [
        (np.array([], dtype=np.uint8), ValueError),
        (np.zeros((4, 4, 2), dtype=np.uint8), ValueError),
        (np.zeros((4, 4), dtype=np.float32), TypeError),
        ("not an image", TypeError),
    ],
)
def test_equalize_histogram_rejects_invalid_images(image, error_type):
    with pytest.raises(error_type):
        equalize_histogram(image)
