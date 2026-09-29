import numpy as np
import pytest

from processing.preprocessing.clahe import apply_clahe


def test_clahe_supports_grayscale_images_with_default_parameters():
    image = np.tile(np.arange(100, 116, dtype=np.uint8), (16, 1))

    result = apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))


def test_clahe_supports_custom_parameters():
    image = np.tile(np.arange(100, 116, dtype=np.uint8), (16, 1))

    result = apply_clahe(image, clip_limit=3.5, tile_grid_size=(4, 4))

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_clahe_supports_bgr_images_without_channel_loss():
    image = np.full((32, 32, 3), (60, 90, 120), dtype=np.uint8)
    image[:, :16] += 15

    result = apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3
    assert np.all((result >= 0) & (result <= 255))


def test_clahe_preserves_bgra_alpha_channel():
    image = np.full((32, 32, 4), (60, 90, 120, 0), dtype=np.uint8)
    image[:, :16, :3] += 15
    image[:, 16:, 3] = 255

    result = apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_clahe_does_not_mutate_input():
    image = np.full((32, 32, 3), (60, 90, 120), dtype=np.uint8)
    image[:, :16] += 15
    original = image.copy()

    apply_clahe(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize("clip_limit", [0, -1, float("inf"), float("nan"), True])
def test_clahe_rejects_invalid_clip_limit(clip_limit):
    image = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_clahe(image, clip_limit=clip_limit)


@pytest.mark.parametrize(
    "tile_grid_size",
    [(0, 8), (8, 0), (-1, 8), (8,), [8, 8], (8.0, 8), (True, 8)],
)
def test_clahe_rejects_invalid_tile_grid_size(tile_grid_size):
    image = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_clahe(image, tile_grid_size=tile_grid_size)
