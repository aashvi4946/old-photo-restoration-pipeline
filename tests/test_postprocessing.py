import numpy as np
import pytest

from processing.postprocessing.black_white import convert_to_black_and_white
from processing.postprocessing.brightness import adjust_brightness
from processing.postprocessing.color_correction import apply_color_correction
from processing.postprocessing.contrast import adjust_contrast
from processing.postprocessing.edge_enhancement import enhance_edges
from processing.postprocessing.gamma import adjust_gamma
from processing.postprocessing.sharpen import sharpen_image
from processing.postprocessing.vintage import apply_vintage_style


def test_brightness_adjustment_clips_values_and_does_not_mutate_input():
    image = np.array([[0, 100, 255]], dtype=np.uint8)
    original = image.copy()

    result = adjust_brightness(image, 50)

    assert np.array_equal(result, np.array([[50, 150, 255]], dtype=np.uint8))
    assert np.array_equal(image, original)


def test_brightness_and_contrast_preserve_bgra_alpha():
    image = np.full((3, 3, 4), (100, 120, 140, 88), dtype=np.uint8)
    original = image.copy()

    brightened = adjust_brightness(image, 20)
    contrasted = adjust_contrast(image, 1.5)

    assert np.array_equal(brightened[:, :, 3], original[:, :, 3])
    assert np.array_equal(contrasted[:, :, 3], original[:, :, 3])


@pytest.mark.parametrize("value", [-256, 256, float("inf"), "1", True])
def test_brightness_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        adjust_brightness(np.zeros((3, 3), dtype=np.uint8), value)


def test_contrast_adjustment_expands_distance_from_mid_gray():
    image = np.array([[100, 128, 156]], dtype=np.uint8)

    result = adjust_contrast(image, 2.0)

    assert result[0, 0] < image[0, 0]
    assert result[0, 2] > image[0, 2]


@pytest.mark.parametrize("factor", [0, -1, float("nan"), "1", True])
def test_contrast_rejects_invalid_factors(factor):
    with pytest.raises(ValueError):
        adjust_contrast(np.zeros((3, 3), dtype=np.uint8), factor)


def test_gamma_above_one_brightens_midtones_and_preserves_alpha():
    image = np.full((3, 3, 4), (64, 64, 64, 123), dtype=np.uint8)
    original = image.copy()

    result = adjust_gamma(image, 2.0)

    assert result[0, 0, 0] > image[0, 0, 0]
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize("gamma", [0, -1, float("inf"), "1", True])
def test_gamma_rejects_invalid_values(gamma):
    with pytest.raises(ValueError):
        adjust_gamma(np.zeros((3, 3), dtype=np.uint8), gamma)


def test_color_correction_applies_bgr_gains_and_preserves_bgra_alpha():
    image = np.full((3, 3, 4), (50, 100, 150, 77), dtype=np.uint8)
    original = image.copy()

    result = apply_color_correction(image, 2.0, 0.5, 1.0)

    assert np.array_equal(result[0, 0], np.array([100, 50, 150, 77], dtype=np.uint8))
    assert np.array_equal(image, original)


def test_color_correction_returns_an_independent_grayscale_copy():
    image = np.full((3, 3), 100, dtype=np.uint8)

    result = apply_color_correction(image, 2.0, 1.0, 0.5)

    assert np.array_equal(result, image)
    assert result is not image


@pytest.mark.parametrize("gain", [0, -1, float("nan"), "1", True])
def test_color_correction_rejects_invalid_gains(gain):
    with pytest.raises(ValueError):
        apply_color_correction(np.zeros((3, 3, 3), dtype=np.uint8), red_gain=gain)


def test_sharpen_and_edge_enhancement_support_bgra_without_alpha_loss():
    image = np.zeros((9, 9, 4), dtype=np.uint8)
    image[4, 4] = (80, 140, 200, 99)
    image[:, :, 3] = np.arange(9, dtype=np.uint8)
    original = image.copy()

    sharpened = sharpen_image(image, 1.0)
    enhanced = enhance_edges(image, 0.5)

    assert sharpened.shape == image.shape
    assert enhanced.shape == image.shape
    assert np.array_equal(sharpened[:, :, 3], original[:, :, 3])
    assert np.array_equal(enhanced[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize("operation", [sharpen_image, enhance_edges])
@pytest.mark.parametrize("value", [-1, float("inf"), "1", True])
def test_detail_finish_operations_reject_invalid_strength(operation, value):
    with pytest.raises(ValueError):
        operation(np.zeros((3, 3), dtype=np.uint8), value)


def test_black_and_white_removes_color_and_retains_bgra_alpha():
    image = np.full((3, 3, 4), (50, 100, 200, 88), dtype=np.uint8)
    original = image.copy()

    result = convert_to_black_and_white(image)

    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


def test_vintage_style_adds_warmth_and_preserves_alpha_and_input():
    image = np.full((3, 3, 4), (80, 100, 120, 200), dtype=np.uint8)
    original = image.copy()

    result = apply_vintage_style(image, 1.0)

    assert result[0, 0, 2] > result[0, 0, 0]
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize("intensity", [-0.1, 1.1, float("nan"), "1", True])
def test_vintage_rejects_invalid_intensity(intensity):
    with pytest.raises(ValueError):
        apply_vintage_style(np.zeros((3, 3), dtype=np.uint8), intensity)
