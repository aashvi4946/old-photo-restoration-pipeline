"""GUI-independent manual pipeline construction and execution for Advanced mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processing._image_utils import validate_uint8_image
from processing.postprocessing.black_white import convert_to_black_and_white
from processing.postprocessing.brightness import adjust_brightness
from processing.postprocessing.color_correction import apply_color_correction
from processing.postprocessing.contrast import adjust_contrast
from processing.postprocessing.edge_enhancement import enhance_edges
from processing.postprocessing.gamma import adjust_gamma
from processing.postprocessing.sharpen import sharpen_image
from processing.postprocessing.threshold import apply_adaptive_threshold
from processing.postprocessing.vintage import apply_vintage_style
from processing.preprocessing.clahe import apply_clahe
from processing.preprocessing.histogram import equalize_histogram
from processing.preprocessing.normalization import normalize_image
from processing.preprocessing.resize import resize_image
from processing.preprocessing.white_balance import auto_white_balance
from processing.restoration.deblur import (
    apply_lucy_richardson_deconvolution,
    apply_unsharp_mask,
    apply_wiener_deconvolution,
    create_motion_psf,
)
from processing.restoration.denoise import (
    apply_bilateral_denoise,
    apply_gaussian_denoise,
    apply_median_denoise,
    apply_non_local_means_denoise,
)
from processing.restoration.inpaint import apply_inpainting


@dataclass(frozen=True)
class ParameterSpec:
    """One editable Advanced-mode operation parameter."""

    name: str
    label: str
    kind: str
    default: object
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class OperationDefinition:
    """Metadata for an operation exposed by the Advanced operation browser."""

    identifier: str
    label: str
    category: str
    parameters: tuple[ParameterSpec, ...] = ()


@dataclass(frozen=True)
class PipelineStep:
    """A configured operation in an ordered manual processing pipeline."""

    operation: str
    parameters: dict[str, object]


def _float(name, label, default, minimum, maximum, step=0.1):
    return ParameterSpec(name, label, "float", default, minimum, maximum, step)


def _integer(name, label, default, minimum, maximum, step=1):
    return ParameterSpec(name, label, "int", default, minimum, maximum, step)


OPERATION_DEFINITIONS = (
    OperationDefinition("histogram", "Histogram Equalization", "Prepare"),
    OperationDefinition(
        "clahe", "CLAHE", "Prepare", (_float("clip_limit", "Clip Limit", 2.0, 0.1, 100.0), _integer("tile_width", "Tile Width", 8, 1, 256), _integer("tile_height", "Tile Height", 8, 1, 256)),
    ),
    OperationDefinition("white_balance", "Auto White Balance", "Prepare"),
    OperationDefinition(
        "normalize", "Normalize", "Prepare", (_integer("new_min", "New Minimum", 0, 0, 254), _integer("new_max", "New Maximum", 255, 1, 255)),
    ),
    OperationDefinition(
        "resize", "Resize", "Prepare", (_integer("width", "Width", 1000, 1, 100000), _integer("height", "Height", 1000, 1, 100000)),
    ),
    OperationDefinition(
        "gaussian", "Gaussian Denoise", "Restore", (_integer("kernel_size", "Kernel Size", 5, 3, 99, 2), _float("sigma", "Sigma", 0.0, 0.0, 1000.0)),
    ),
    OperationDefinition("median", "Median Denoise", "Restore", (_integer("kernel_size", "Kernel Size", 3, 3, 99, 2),)),
    OperationDefinition(
        "bilateral", "Bilateral Denoise", "Restore", (_integer("diameter", "Diameter", 9, 1, 99), _float("sigma_color", "Color Sigma", 75.0, 0.1, 1000.0), _float("sigma_space", "Space Sigma", 75.0, 0.1, 1000.0)),
    ),
    OperationDefinition(
        "non_local_means", "Non-Local Means", "Restore", (_float("h", "Strength", 10.0, 0.1, 1000.0), _float("h_color", "Color Strength", 10.0, 0.1, 1000.0), _integer("template_window_size", "Template Window", 7, 3, 31, 2), _integer("search_window_size", "Search Window", 21, 3, 51, 2)),
    ),
    OperationDefinition(
        "unsharp", "Unsharp Mask", "Restore", (_float("radius", "Radius", 1.0, 0.1, 100.0), _float("amount", "Amount", 1.0, 0.1, 10.0), _integer("threshold", "Threshold", 0, 0, 255)),
    ),
    OperationDefinition(
        "wiener", "Wiener Deconvolution", "Restore", (_integer("psf_length", "PSF Length", 9, 3, 31, 2), _float("psf_angle", "PSF Angle", 0.0, -360.0, 360.0), _float("balance", "Balance", 0.01, 0.001, 1.0, 0.001)),
    ),
    OperationDefinition(
        "lucy_richardson", "Lucy–Richardson", "Restore", (_integer("psf_length", "PSF Length", 9, 3, 31, 2), _float("psf_angle", "PSF Angle", 0.0, -360.0, 360.0), _integer("iterations", "Iterations", 10, 1, 100)),
    ),
    OperationDefinition(
        "inpaint", "Inpainting (Repair Mask)", "Restore", (ParameterSpec("method", "Method", "choice", "telea", choices=(("Telea", "telea"), ("Navier-Stokes", "navier_stokes"))), _float("radius", "Radius", 3.0, 0.1, 100.0, 0.5)),
    ),
    OperationDefinition("sharpen", "Sharpen", "Finish", (_float("amount", "Amount", 1.0, 0.0, 3.0),)),
    OperationDefinition("edge_enhancement", "Edge Enhancement", "Finish", (_float("strength", "Strength", 0.5, 0.0, 3.0),)),
    OperationDefinition("brightness", "Brightness", "Finish", (_integer("value", "Value", 0, -255, 255),)),
    OperationDefinition("contrast", "Contrast", "Finish", (_float("factor", "Factor", 1.0, 0.1, 3.0),)),
    OperationDefinition("gamma", "Gamma", "Finish", (_float("gamma", "Gamma", 1.0, 0.1, 5.0),)),
    OperationDefinition(
        "color_correction", "Color Correction", "Finish", (_float("blue_gain", "Blue Gain", 1.0, 0.1, 3.0), _float("green_gain", "Green Gain", 1.0, 0.1, 3.0), _float("red_gain", "Red Gain", 1.0, 0.1, 3.0)),
    ),
    OperationDefinition(
        "adaptive_threshold", "Adaptive Threshold", "Finish", (_integer("block_size", "Block Size", 11, 3, 255, 2), _float("constant", "Constant", 2.0, -255.0, 255.0), ParameterSpec("method", "Method", "choice", "gaussian", choices=(("Gaussian", "gaussian"), ("Mean", "mean")))),
    ),
    OperationDefinition("black_white", "B&W", "Finish"),
    OperationDefinition("vintage", "Vintage", "Finish", (_float("intensity", "Intensity", 0.35, 0.0, 1.0, 0.05),)),
)
_DEFINITIONS_BY_ID = {definition.identifier: definition for definition in OPERATION_DEFINITIONS}


class ProcessingPipeline:
    """Ordered, manually configured pipeline that contains no GUI dependencies."""

    def __init__(self) -> None:
        self._steps: list[PipelineStep] = []

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        return tuple(self._steps)

    @property
    def is_empty(self) -> bool:
        return not self._steps

    def add(self, operation: str) -> PipelineStep:
        definition = self.definition_for(operation)
        step = PipelineStep(operation, {spec.name: spec.default for spec in definition.parameters})
        self._steps.append(step)
        return step

    def remove(self, index: int) -> PipelineStep:
        return self._steps.pop(self._validate_index(index))

    def move(self, index: int, offset: int) -> int:
        index = self._validate_index(index)
        target = index + offset
        if target < 0 or target >= len(self._steps):
            raise ValueError("Pipeline step cannot be moved beyond the pipeline bounds.")
        self._steps[index], self._steps[target] = self._steps[target], self._steps[index]
        return target

    def update_parameters(self, index: int, parameters: dict[str, object]) -> None:
        index = self._validate_index(index)
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary.")
        definition = self.definition_for(self._steps[index].operation)
        expected = {spec.name for spec in definition.parameters}
        if set(parameters) != expected:
            raise ValueError("parameters must match the selected operation's parameter set.")
        self._steps[index] = PipelineStep(self._steps[index].operation, dict(parameters))

    def clear(self) -> None:
        self._steps.clear()

    def execute(self, image: np.ndarray, repair_mask: np.ndarray | None = None) -> np.ndarray:
        """Execute all configured steps on a copy of the supplied image."""

        validate_uint8_image(image)
        if self.is_empty:
            raise ValueError("Add at least one operation to the pipeline.")
        result = image.copy()
        for step in self._steps:
            result = self._execute_step(result, step, repair_mask)
        return result

    @staticmethod
    def definition_for(operation: str) -> OperationDefinition:
        try:
            return _DEFINITIONS_BY_ID[operation]
        except KeyError as error:
            raise ValueError(f"Unknown pipeline operation: {operation}") from error

    def _execute_step(self, image, step, repair_mask):
        params = step.parameters
        operations = {
            "histogram": lambda: equalize_histogram(image),
            "clahe": lambda: apply_clahe(image, params["clip_limit"], (params["tile_width"], params["tile_height"])),
            "white_balance": lambda: auto_white_balance(image),
            "normalize": lambda: normalize_image(image, params["new_min"], params["new_max"]),
            "resize": lambda: resize_image(image, params["width"], params["height"]),
            "gaussian": lambda: apply_gaussian_denoise(image, **params),
            "median": lambda: apply_median_denoise(image, **params),
            "bilateral": lambda: apply_bilateral_denoise(image, **params),
            "non_local_means": lambda: apply_non_local_means_denoise(image, **params),
            "unsharp": lambda: apply_unsharp_mask(image, **params),
            "wiener": lambda: apply_wiener_deconvolution(image, create_motion_psf(params["psf_length"], params["psf_angle"]), params["balance"]),
            "lucy_richardson": lambda: apply_lucy_richardson_deconvolution(image, create_motion_psf(params["psf_length"], params["psf_angle"]), params["iterations"]),
            "inpaint": lambda: apply_inpainting(image, repair_mask, **params),
            "sharpen": lambda: sharpen_image(image, **params),
            "edge_enhancement": lambda: enhance_edges(image, **params),
            "brightness": lambda: adjust_brightness(image, **params),
            "contrast": lambda: adjust_contrast(image, **params),
            "gamma": lambda: adjust_gamma(image, **params),
            "color_correction": lambda: apply_color_correction(image, **params),
            "adaptive_threshold": lambda: apply_adaptive_threshold(image, **params),
            "black_white": lambda: convert_to_black_and_white(image),
            "vintage": lambda: apply_vintage_style(image, **params),
        }
        return operations[step.operation]()

    def _validate_index(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self._steps):
            raise ValueError("index must identify an existing pipeline step.")
        return index
