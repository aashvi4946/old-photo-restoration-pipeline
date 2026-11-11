import numpy as np
import pytest

from processing.restoration.deblur import (
    apply_lucy_richardson_deconvolution,
    apply_unsharp_mask,
    apply_wiener_deconvolution,
    create_motion_psf,
)


def test_unsharp_mask_enhances_a_local_detail_signal():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 160

    result = apply_unsharp_mask(image, radius=1.0, amount=0.5)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result[4, 4] > image[4, 4]
    assert np.all((result >= 0) & (result <= 255))


def test_unsharp_mask_supports_bgr_images_without_channel_loss():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[4, 4] = (50, 100, 160)

    result = apply_unsharp_mask(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3


def test_unsharp_mask_preserves_bgra_alpha_channel():
    image = np.zeros((9, 9, 4), dtype=np.uint8)
    image[4, 4] = (50, 100, 160, 123)
    image[:, :, 3] = np.arange(9, dtype=np.uint8)
    alpha = image[:, :, 3].copy()

    result = apply_unsharp_mask(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], alpha)


def test_unsharp_mask_threshold_can_leave_low_amplitude_detail_unchanged():
    image = np.full((9, 9), 100, dtype=np.uint8)
    image[4, 4] = 101

    result = apply_unsharp_mask(image, threshold=10)

    assert np.array_equal(result, image)


def test_unsharp_mask_does_not_mutate_input():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 160
    original = image.copy()

    apply_unsharp_mask(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize("radius", [0, -1, float("inf"), float("nan"), "1", True])
def test_unsharp_mask_rejects_invalid_radius(radius):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_unsharp_mask(image, radius=radius)


@pytest.mark.parametrize("amount", [0, -1, float("inf"), float("nan"), "1", True])
def test_unsharp_mask_rejects_invalid_amount(amount):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_unsharp_mask(image, amount=amount)


@pytest.mark.parametrize("threshold", [-1, 256, 1.0, "1", True])
def test_unsharp_mask_rejects_invalid_threshold(threshold):
    image = np.zeros((9, 9), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_unsharp_mask(image, threshold=threshold)


def test_create_motion_psf_returns_normalized_square_kernel():
    psf = create_motion_psf(length=9, angle=45.0)

    assert psf.shape == (9, 9)
    assert psf.dtype == np.float64
    assert psf.sum() == pytest.approx(1.0)
    assert np.all(psf >= 0)


@pytest.mark.parametrize("length", [0, -1, 2, 4, 3.0, True])
def test_create_motion_psf_rejects_invalid_length(length):
    with pytest.raises(ValueError):
        create_motion_psf(length=length)


@pytest.mark.parametrize("angle", [float("inf"), float("nan"), "0", True])
def test_create_motion_psf_rejects_invalid_angle(angle):
    with pytest.raises(ValueError):
        create_motion_psf(angle=angle)


def test_wiener_deconvolution_supports_grayscale_images():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[6:10, 6:10] = 180
    psf = create_motion_psf(length=3)

    result = apply_wiener_deconvolution(image, psf, balance=0.1)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))


def test_wiener_deconvolution_supports_bgr_images_without_channel_loss():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[6:10, 6:10] = (50, 100, 180)
    psf = create_motion_psf(length=3)

    result = apply_wiener_deconvolution(image, psf, balance=0.1)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3


def test_wiener_deconvolution_preserves_bgra_alpha_and_input():
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[6:10, 6:10] = (50, 100, 180, 123)
    image[:, :, 3] = np.arange(16, dtype=np.uint8)
    original = image.copy()
    psf = create_motion_psf(length=3)

    result = apply_wiener_deconvolution(image, psf, balance=0.1)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


def test_wiener_deconvolution_accepts_an_identity_psf():
    image = np.full((8, 8), 100, dtype=np.uint8)

    result = apply_wiener_deconvolution(image, np.array([[1.0]]), balance=1e-10)

    assert np.array_equal(result, image)


@pytest.mark.parametrize(
    "psf, error_type",
    [
        (np.array([]), ValueError),
        (np.zeros((3, 3)), ValueError),
        (np.array([1, 2, 3]), ValueError),
        ("not a psf", TypeError),
    ],
)
def test_wiener_deconvolution_rejects_invalid_psf(psf, error_type):
    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(error_type):
        apply_wiener_deconvolution(image, psf)


@pytest.mark.parametrize("balance", [0, -1, float("inf"), float("nan"), "1", True])
def test_wiener_deconvolution_rejects_invalid_balance(balance):
    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_wiener_deconvolution(image, np.array([[1.0]]), balance=balance)


def test_lucy_richardson_supports_grayscale_images():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[6:10, 6:10] = 180
    psf = create_motion_psf(length=3)

    result = apply_lucy_richardson_deconvolution(image, psf, iterations=3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))


def test_lucy_richardson_supports_bgr_images_without_channel_loss():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[6:10, 6:10] = (50, 100, 180)

    result = apply_lucy_richardson_deconvolution(
        image,
        create_motion_psf(length=3),
        iterations=3,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.shape[2] == 3


def test_lucy_richardson_preserves_bgra_alpha_and_input():
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[6:10, 6:10] = (50, 100, 180, 123)
    image[:, :, 3] = np.arange(16, dtype=np.uint8)
    original = image.copy()

    result = apply_lucy_richardson_deconvolution(
        image,
        create_motion_psf(length=3),
        iterations=3,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


def test_lucy_richardson_accepts_an_identity_psf():
    image = np.full((8, 8), 100, dtype=np.uint8)

    result = apply_lucy_richardson_deconvolution(
        image,
        np.array([[1.0]]),
        iterations=3,
    )

    assert np.array_equal(result, image)


@pytest.mark.parametrize("iterations", [0, -1, 1.0, "1", True])
def test_lucy_richardson_rejects_invalid_iterations(iterations):
    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_lucy_richardson_deconvolution(
            image,
            np.array([[1.0]]),
            iterations=iterations,
        )


def test_deconvolution_rejects_negative_psf_values():
    image = np.zeros((8, 8), dtype=np.uint8)
    psf = np.array([[1.0, -0.5], [0.0, 0.5]])

    with pytest.raises(ValueError):
        apply_wiener_deconvolution(image, psf)
