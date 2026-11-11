"""Classical deblurring operations for grayscale, BGR, and BGRA images."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def apply_unsharp_mask(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    """Enhance local detail by adding a scaled high-frequency image mask.

    ``radius`` controls the Gaussian blur used to derive the mask, ``amount``
    controls sharpening strength, and ``threshold`` prevents low-amplitude
    detail from being sharpened. BGRA alpha is kept unchanged.
    """

    validate_uint8_image(image)
    _validate_unsharp_parameters(radius, amount, threshold)

    color_image, alpha = split_alpha(image)
    blurred = cv2.GaussianBlur(color_image, (0, 0), sigmaX=float(radius))
    detail = color_image.astype(np.float64) - blurred.astype(np.float64)

    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0

    sharpened = np.clip(
        np.rint(color_image.astype(np.float64) + amount * detail),
        0,
        255,
    ).astype(np.uint8)

    return restore_alpha(sharpened, alpha)


def create_motion_psf(length: int = 9, angle: float = 0.0) -> np.ndarray:
    """Create a normalized square motion-blur point spread function.

    ``angle`` is measured in degrees counter-clockwise from the horizontal.
    The returned floating-point kernel sums to one and can be passed directly
    to :func:`apply_wiener_deconvolution`.
    """

    _validate_motion_psf_parameters(length, angle)

    psf = np.zeros((length, length), dtype=np.float64)
    center = (length - 1) / 2.0
    radius = center
    angle_radians = np.deg2rad(angle)
    offset_x = radius * np.cos(angle_radians)
    offset_y = radius * np.sin(angle_radians)

    start = (round(center - offset_x), round(center - offset_y))
    end = (round(center + offset_x), round(center + offset_y))
    cv2.line(psf, start, end, color=1.0, thickness=1, lineType=cv2.LINE_AA)

    return psf / psf.sum()


def apply_wiener_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    balance: float = 0.01,
) -> np.ndarray:
    """Apply frequency-domain Wiener deconvolution using a supplied PSF.

    ``balance`` regularizes the inverse filter. Higher values suppress noise
    amplification at the expense of less aggressive deblurring. The supplied
    PSF is normalized internally and must be a finite, non-empty 2D array.
    BGRA alpha is retained unchanged.
    """

    validate_uint8_image(image)
    normalized_psf = _validate_and_normalize_psf(psf)
    _validate_wiener_balance(balance)

    color_image, alpha = split_alpha(image)
    if color_image.ndim == 2:
        restored = _wiener_channel(color_image, normalized_psf, balance)
    else:
        restored = np.dstack(
            [
                _wiener_channel(color_image[:, :, channel], normalized_psf, balance)
                for channel in range(3)
            ]
        )

    return restore_alpha(restored, alpha)


def apply_lucy_richardson_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 10,
) -> np.ndarray:
    """Apply iterative Lucy–Richardson deconvolution using a supplied PSF.

    More iterations can recover more apparent detail but may amplify noise and
    ringing artifacts. This deterministic classical implementation processes
    each BGR channel independently and keeps BGRA alpha unchanged.
    """

    validate_uint8_image(image)
    normalized_psf = _validate_and_normalize_psf(psf)
    _validate_lucy_richardson_iterations(iterations)

    color_image, alpha = split_alpha(image)
    if color_image.ndim == 2:
        restored = _lucy_richardson_channel(
            color_image,
            normalized_psf,
            iterations,
        )
    else:
        restored = np.dstack(
            [
                _lucy_richardson_channel(
                    color_image[:, :, channel],
                    normalized_psf,
                    iterations,
                )
                for channel in range(3)
            ]
        )

    return restore_alpha(restored, alpha)


def _wiener_channel(
    channel: np.ndarray,
    psf: np.ndarray,
    balance: float,
) -> np.ndarray:
    """Restore one image channel in the Fourier domain."""

    transfer_function = np.fft.fft2(np.fft.ifftshift(psf), s=channel.shape)
    observed = np.fft.fft2(channel.astype(np.float64))
    inverse_filter = np.conj(transfer_function) / (
        np.abs(transfer_function) ** 2 + balance
    )
    restored = np.fft.ifft2(observed * inverse_filter).real

    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


def _lucy_richardson_channel(
    channel: np.ndarray,
    psf: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Run Lucy–Richardson updates for one uint8 image channel."""

    observed = channel.astype(np.float64) / 255.0
    estimate = observed.copy()
    flipped_psf = np.flip(psf)
    epsilon = np.finfo(np.float64).eps

    for _ in range(iterations):
        blurred_estimate = cv2.filter2D(
            estimate,
            ddepth=-1,
            kernel=psf,
            borderType=cv2.BORDER_REFLECT,
        )
        relative_blur = observed / np.maximum(blurred_estimate, epsilon)
        estimate *= cv2.filter2D(
            relative_blur,
            ddepth=-1,
            kernel=flipped_psf,
            borderType=cv2.BORDER_REFLECT,
        )
        estimate = np.maximum(estimate, 0.0)

    return np.clip(np.rint(estimate * 255.0), 0, 255).astype(np.uint8)


def _validate_unsharp_parameters(
    radius: float,
    amount: float,
    threshold: int,
) -> None:
    """Validate Unsharp Mask settings before image processing."""

    for name, value, minimum in (("radius", radius, 0), ("amount", amount, 0)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= minimum
        ):
            raise ValueError(f"{name} must be a finite number greater than zero.")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 <= threshold <= 255
    ):
        raise ValueError("threshold must be an integer from 0 to 255.")


def _validate_motion_psf_parameters(length: int, angle: float) -> None:
    """Validate motion-PSF settings."""

    if (
        isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or length % 2 == 0
    ):
        raise ValueError("length must be a positive odd integer.")

    if (
        isinstance(angle, bool)
        or not isinstance(angle, Real)
        or not np.isfinite(angle)
    ):
        raise ValueError("angle must be a finite number.")


def _validate_and_normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Validate and normalize a 2D point spread function."""

    if not isinstance(psf, np.ndarray):
        raise TypeError("psf must be a NumPy array.")

    if psf.ndim != 2 or psf.size == 0:
        raise ValueError("psf must be a non-empty 2D array.")

    if not np.issubdtype(psf.dtype, np.number) or not np.all(np.isfinite(psf)):
        raise ValueError("psf must contain finite numeric values.")

    if np.any(psf < 0):
        raise ValueError("psf must not contain negative values.")

    psf_sum = float(psf.sum())
    if psf_sum <= 0:
        raise ValueError("psf must have a positive sum.")

    return psf.astype(np.float64, copy=True) / psf_sum


def _validate_wiener_balance(balance: float) -> None:
    """Validate the Wiener regularization balance."""

    if (
        isinstance(balance, bool)
        or not isinstance(balance, Real)
        or not np.isfinite(balance)
        or balance <= 0
    ):
        raise ValueError("balance must be a finite number greater than zero.")


def _validate_lucy_richardson_iterations(iterations: int) -> None:
    """Validate the requested number of deconvolution updates."""

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer.")
