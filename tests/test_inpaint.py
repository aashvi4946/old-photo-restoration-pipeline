import numpy as np
import pytest

from processing.restoration.inpaint import apply_inpainting


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_inpainting_repairs_a_marked_grayscale_pixel(method):
    image = np.full((20, 20), 200, dtype=np.uint8)
    image[10, 10] = 0
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255

    result = apply_inpainting(image, mask, method=method)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result[10, 10] > image[10, 10]


def test_inpainting_preserves_bgra_alpha_and_input():
    image = np.full((20, 20, 4), (50, 100, 200, 123), dtype=np.uint8)
    image[10, 10, :3] = 0
    image[:, :, 3] = np.arange(20, dtype=np.uint8)
    original = image.copy()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255

    result = apply_inpainting(image, mask)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "mask, error_type",
    [
        (np.zeros((20, 20), dtype=np.uint8), ValueError),
        (np.ones((10, 10), dtype=np.uint8), ValueError),
        (np.array([], dtype=np.uint8), ValueError),
        ("not a mask", TypeError),
    ],
)
def test_inpainting_rejects_invalid_masks(mask, error_type):
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(error_type):
        apply_inpainting(image, mask)


@pytest.mark.parametrize("method", ["unknown", "TELEA", 1])
def test_inpainting_rejects_invalid_methods(method):
    image = np.zeros((20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255

    with pytest.raises(ValueError):
        apply_inpainting(image, mask, method=method)


@pytest.mark.parametrize("radius", [0, -1, float("inf"), float("nan"), "3", True])
def test_inpainting_rejects_invalid_radius(radius):
    image = np.zeros((20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255

    with pytest.raises(ValueError):
        apply_inpainting(image, mask, radius=radius)
