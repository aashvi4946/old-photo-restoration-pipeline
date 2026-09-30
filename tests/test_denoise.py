import numpy as np
import pytest

from processing.restoration.denoise import (
    apply_bilateral_denoise,
    apply_gaussian_denoise,
    apply_median_denoise,
    apply_non_local_means_denoise,
)


def test_gaussian_denoise_supports_grayscale_images():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 255

    result = apply_gaussian_denoise(image, kernel_size=3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert 0 < result[4, 4] < 255
    assert result[4, 3] > 0


def test_gaussian_denoise_supports_bgr_images_without_channel_loss():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[4, 4] = (50, 150, 255)

    result = apply_gaussian_denoise(image, kernel_size=3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3
    assert np.all((result >= 0) & (result <= 255))


def test_gaussian_denoise_preserves_bgra_alpha_channel():
    image = np.zeros((9, 9, 4), dtype=np.uint8)
    image[4, 4] = (50, 150, 255, 123)
    image[:, :, 3] = np.arange(9, dtype=np.uint8)
    alpha = image[:, :, 3].copy()

    result = apply_gaussian_denoise(image, kernel_size=3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], alpha)


def test_gaussian_denoise_does_not_mutate_input():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 255
    original = image.copy()

    apply_gaussian_denoise(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize("kernel_size", [0, -1, 2, 4, 3.0, True])
def test_gaussian_denoise_rejects_invalid_kernel_size(kernel_size):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_gaussian_denoise(image, kernel_size=kernel_size)


@pytest.mark.parametrize("sigma", [-1, float("inf"), float("nan"), "1", True])
def test_gaussian_denoise_rejects_invalid_sigma(sigma):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_gaussian_denoise(image, sigma=sigma)


def test_median_denoise_removes_isolated_impulse_noise():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 255

    result = apply_median_denoise(image, kernel_size=3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result[4, 4] == 0


def test_median_denoise_preserves_bgra_alpha_channel():
    image = np.zeros((9, 9, 4), dtype=np.uint8)
    image[4, 4] = (50, 150, 255, 123)
    image[:, :, 3] = np.arange(9, dtype=np.uint8)
    alpha = image[:, :, 3].copy()

    result = apply_median_denoise(image, kernel_size=3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], alpha)


@pytest.mark.parametrize("kernel_size", [0, -1, 2, 4, 3.0, True])
def test_median_denoise_rejects_invalid_kernel_size(kernel_size):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_median_denoise(image, kernel_size=kernel_size)


def test_bilateral_denoise_supports_bgr_images_without_channel_loss():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[4, 4] = (50, 150, 255)

    result = apply_bilateral_denoise(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3
    assert np.all((result >= 0) & (result <= 255))


def test_bilateral_denoise_preserves_bgra_alpha_channel_and_input():
    image = np.zeros((9, 9, 4), dtype=np.uint8)
    image[4, 4] = (50, 150, 255, 123)
    image[:, :, 3] = np.arange(9, dtype=np.uint8)
    original = image.copy()

    result = apply_bilateral_denoise(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize("diameter", [0, -1, 3.0, True])
def test_bilateral_denoise_rejects_invalid_diameter(diameter):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_bilateral_denoise(image, diameter=diameter)


@pytest.mark.parametrize("parameter", ["sigma_color", "sigma_space"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "1", True])
def test_bilateral_denoise_rejects_invalid_sigmas(parameter, value):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_bilateral_denoise(image, **{parameter: value})


def test_non_local_means_denoise_supports_grayscale_images():
    random = np.random.default_rng(1234)
    image = random.integers(0, 256, size=(32, 32), dtype=np.uint8)

    result = apply_non_local_means_denoise(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))


def test_non_local_means_denoise_supports_bgr_images_without_channel_loss():
    random = np.random.default_rng(1234)
    image = random.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    result = apply_non_local_means_denoise(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3


def test_non_local_means_denoise_preserves_bgra_alpha_and_input():
    random = np.random.default_rng(1234)
    image = random.integers(0, 256, size=(32, 32, 4), dtype=np.uint8)
    original = image.copy()

    result = apply_non_local_means_denoise(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize("parameter", ["h", "h_color"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "1", True])
def test_non_local_means_rejects_invalid_strengths(parameter, value):
    image = np.zeros((32, 32), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_non_local_means_denoise(image, **{parameter: value})


@pytest.mark.parametrize("parameter", ["template_window_size", "search_window_size"])
@pytest.mark.parametrize("value", [0, -1, 2, 4, 3.0, True])
def test_non_local_means_rejects_invalid_window_sizes(parameter, value):
    image = np.zeros((32, 32), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_non_local_means_denoise(image, **{parameter: value})
