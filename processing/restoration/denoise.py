"""Classical denoising operations for grayscale, BGR, and BGRA images."""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np

from processing._image_utils import restore_alpha, split_alpha, validate_uint8_image


def apply_gaussian_denoise(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Reduce fine noise with a Gaussian blur while preserving alpha.

    A Gaussian filter is a simple, deterministic denoising method. It can also
    soften edges, so callers should use the smallest effective kernel.
    ``sigma=0`` delegates sigma selection to OpenCV from the kernel size.
    """

    validate_uint8_image(image)
    _validate_odd_kernel_size(kernel_size)
    _validate_gaussian_sigma(sigma)

    color_image, alpha = split_alpha(image)
    denoised = cv2.GaussianBlur(
        color_image,
        (kernel_size, kernel_size),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    return restore_alpha(denoised, alpha)


def apply_median_denoise(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Reduce impulse noise with a median filter while preserving alpha."""

    validate_uint8_image(image)
    _validate_odd_kernel_size(kernel_size)

    color_image, alpha = split_alpha(image)
    denoised = cv2.medianBlur(color_image, kernel_size)

    return restore_alpha(denoised, alpha)


def apply_bilateral_denoise(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Reduce noise while preserving edges with a bilateral filter.

    Larger sigma values increase the influence of more distant colors and
    pixels, which can produce stronger smoothing. The operation is explicit;
    it is never selected automatically in this phase.
    """

    validate_uint8_image(image)
    _validate_bilateral_parameters(diameter, sigma_color, sigma_space)

    color_image, alpha = split_alpha(image)
    denoised = cv2.bilateralFilter(
        color_image,
        diameter,
        float(sigma_color),
        float(sigma_space),
    )

    return restore_alpha(denoised, alpha)


def apply_non_local_means_denoise(
    image: np.ndarray,
    h: float = 10.0,
    h_color: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """Reduce noise with classical Non-Local Means while preserving alpha.

    ``h`` controls luminance filtering strength and ``h_color`` controls color
    filtering strength for BGR/BGRA images. Larger values remove more noise but
    may also remove fine detail. This operation can be relatively expensive on
    large images and is only run when explicitly requested.
    """

    validate_uint8_image(image)
    _validate_non_local_means_parameters(
        h,
        h_color,
        template_window_size,
        search_window_size,
    )

    color_image, alpha = split_alpha(image)
    if color_image.ndim == 2:
        denoised = cv2.fastNlMeansDenoising(
            color_image,
            None,
            float(h),
            template_window_size,
            search_window_size,
        )
    else:
        denoised = cv2.fastNlMeansDenoisingColored(
            color_image,
            None,
            float(h),
            float(h_color),
            template_window_size,
            search_window_size,
        )

    return restore_alpha(denoised, alpha)


def _validate_odd_kernel_size(kernel_size: int) -> None:
    """Validate a positive odd kernel dimension."""

    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer.")


def _validate_gaussian_sigma(sigma: float) -> None:
    """Validate the optional Gaussian standard deviation."""

    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, Real)
        or not np.isfinite(sigma)
        or sigma < 0
    ):
        raise ValueError("sigma must be a finite number greater than or equal to zero.")


def _validate_bilateral_parameters(
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> None:
    """Validate bilateral-filter parameters before OpenCV processing."""

    if isinstance(diameter, bool) or not isinstance(diameter, int) or diameter <= 0:
        raise ValueError("diameter must be a positive integer.")

    for name, value in (("sigma_color", sigma_color), ("sigma_space", sigma_space)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite number greater than zero.")


def _validate_non_local_means_parameters(
    h: float,
    h_color: float,
    template_window_size: int,
    search_window_size: int,
) -> None:
    """Validate Non-Local Means parameters before OpenCV processing."""

    for name, value in (("h", h), ("h_color", h_color)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite number greater than zero.")

    for name, value in (
        ("template_window_size", template_window_size),
        ("search_window_size", search_window_size),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 2 == 0
        ):
            raise ValueError(f"{name} must be a positive odd integer.")
