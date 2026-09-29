import numpy as np
import pytest

from processing.preprocessing.normalization import normalize_image


def test_normalize_image_stretches_grayscale_values_to_default_range():
    image = np.array([[100, 105], [110, 115]], dtype=np.uint8)

    result = normalize_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_normalize_image_supports_custom_output_range():
    image = np.array([[100, 105], [110, 115]], dtype=np.uint8)

    result = normalize_image(image, new_min=20, new_max=200)

    assert result.min() == 20
    assert result.max() == 200


def test_normalize_image_handles_constant_images_deterministically():
    image = np.full((10, 10), 128, dtype=np.uint8)

    result = normalize_image(image, new_min=20, new_max=200)

    assert np.array_equal(result, np.full_like(image, 20))


def test_normalize_image_supports_bgr_images_without_channel_loss():
    image = np.full((16, 16, 3), (60, 90, 120), dtype=np.uint8)
    image[:, :8] += 20

    result = normalize_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3
    assert np.all((result >= 0) & (result <= 255))


def test_normalize_image_preserves_bgra_alpha_channel():
    image = np.full((16, 16, 4), (60, 90, 120, 0), dtype=np.uint8)
    image[:, :8, :3] += 20
    image[:, 8:, 3] = 255

    result = normalize_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_normalize_image_does_not_mutate_input():
    image = np.array([[100, 105], [110, 115]], dtype=np.uint8)
    original = image.copy()

    normalize_image(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "new_min, new_max",
    [(-1, 255), (0, 256), (100, 100), (200, 100), (True, 255), (0.0, 255)],
)
def test_normalize_image_rejects_invalid_output_bounds(new_min, new_max):
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        normalize_image(image, new_min=new_min, new_max=new_max)
