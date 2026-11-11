"""Deterministic, explainable Phase 9 Quick Restore orchestration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from models.image_profile import ImageProfile
from models.processing_config import (
    RestorationPlan,
    RestorationStep,
    StrengthConfiguration,
)
from processing._image_utils import validate_uint8_image
from processing.postprocessing.black_white import convert_to_black_and_white
from processing.postprocessing.contrast import adjust_contrast
from processing.postprocessing.gamma import adjust_gamma
from processing.postprocessing.sharpen import sharpen_image
from processing.postprocessing.vintage import apply_vintage_style
from processing.preprocessing.clahe import apply_clahe
from processing.preprocessing.white_balance import auto_white_balance
from processing.restoration.deblur import apply_unsharp_mask
from processing.restoration.denoise import (
    apply_bilateral_denoise,
    apply_non_local_means_denoise,
)


RESTORATION_STRENGTHS = {
    "Light": StrengthConfiguration(
        name="Light",
        noise_medium_parameters={"diameter": 5, "sigma_color": 45.0, "sigma_space": 45.0},
        noise_high_parameters={
            "h": 7.0,
            "h_color": 7.0,
            "template_window_size": 7,
            "search_window_size": 21,
        },
        clahe_clip_limit=1.5,
        contrast_threshold=36.0,
        dark_gamma=1.12,
        bright_gamma=0.92,
        high_blur_unsharp_amount=0.55,
        medium_blur_unsharp_amount=0.0,
        final_sharpen_amount=0.2,
        vintage_intensity=0.2,
    ),
    "Medium": StrengthConfiguration(
        name="Medium",
        noise_medium_parameters={"diameter": 7, "sigma_color": 60.0, "sigma_space": 60.0},
        noise_high_parameters={
            "h": 10.0,
            "h_color": 10.0,
            "template_window_size": 7,
            "search_window_size": 21,
        },
        clahe_clip_limit=2.0,
        contrast_threshold=45.0,
        dark_gamma=1.2,
        bright_gamma=0.88,
        high_blur_unsharp_amount=0.85,
        medium_blur_unsharp_amount=0.45,
        final_sharpen_amount=0.35,
        vintage_intensity=0.35,
    ),
    "Strong": StrengthConfiguration(
        name="Strong",
        noise_medium_parameters={"diameter": 9, "sigma_color": 75.0, "sigma_space": 75.0},
        noise_high_parameters={
            "h": 15.0,
            "h_color": 15.0,
            "template_window_size": 7,
            "search_window_size": 21,
        },
        clahe_clip_limit=3.0,
        contrast_threshold=55.0,
        dark_gamma=1.3,
        bright_gamma=0.82,
        high_blur_unsharp_amount=1.1,
        medium_blur_unsharp_amount=0.7,
        final_sharpen_amount=0.5,
        vintage_intensity=0.5,
    ),
}


class RestorationEngine:
    """Build and execute classical Quick Restore plans without GUI dependencies."""

    _STYLE_NAMES = {"Natural", "Vintage", "B&W"}

    def build_plan(
        self,
        profile: ImageProfile,
        strength: str = "Medium",
        style: str = "Natural",
    ) -> RestorationPlan:
        """Build an ordered plan from a complete image-quality profile."""

        configuration = self._configuration_for(strength)
        self._validate_profile(profile)
        if style not in self._STYLE_NAMES:
            raise ValueError("style must be 'Natural', 'Vintage', or 'B&W'.")

        steps: list[RestorationStep] = []
        skipped: list[str] = []

        self._add_noise_step(profile, configuration, steps, skipped)
        self._add_color_step(profile, steps, skipped)
        self._add_tone_steps(profile, configuration, steps, skipped)
        self._add_deblur_step(profile, configuration, steps, skipped)
        self._add_final_sharpen_step(profile, configuration, steps, skipped)
        self._add_style_step(style, configuration, steps)

        return RestorationPlan(
            strength=configuration.name,
            style=style,
            steps=tuple(steps),
            skipped=tuple(skipped),
        )

    def execute(self, image: np.ndarray, plan: RestorationPlan) -> np.ndarray:
        """Execute a generated plan in order, returning a new valid image."""

        validate_uint8_image(image)
        if not isinstance(plan, RestorationPlan):
            raise TypeError("plan must be a RestorationPlan.")

        processors: dict[str, Callable[..., np.ndarray]] = {
            "bilateral_denoise": apply_bilateral_denoise,
            "non_local_means_denoise": apply_non_local_means_denoise,
            "white_balance": auto_white_balance,
            "gamma": adjust_gamma,
            "clahe": apply_clahe,
            "unsharp": apply_unsharp_mask,
            "sharpen": sharpen_image,
            "contrast": adjust_contrast,
            "vintage": apply_vintage_style,
            "black_white": convert_to_black_and_white,
        }
        result = image.copy()
        for step in plan.steps:
            try:
                processor = processors[step.operation]
            except KeyError as error:
                raise ValueError(f"Unsupported Quick Restore operation: {step.operation}") from error
            result = processor(result, **step.parameters)
        return result

    @staticmethod
    def _configuration_for(strength: str) -> StrengthConfiguration:
        if not isinstance(strength, str) or strength not in RESTORATION_STRENGTHS:
            raise ValueError("strength must be 'Light', 'Medium', or 'Strong'.")
        return RESTORATION_STRENGTHS[strength]

    @staticmethod
    def _validate_profile(profile: ImageProfile) -> None:
        if not isinstance(profile, ImageProfile):
            raise TypeError("profile must be an ImageProfile.")
        if profile.noise_level not in {"Low", "Medium", "High"}:
            raise ValueError("profile must include a valid noise_level.")
        if profile.blur_level not in {"Low", "Medium", "High"}:
            raise ValueError("profile must include a valid blur_level.")
        if profile.color_cast not in {"None", "Blue", "Green", "Red"}:
            raise ValueError("profile must include a valid color_cast.")
        for name, value in (("brightness", profile.brightness), ("contrast", profile.contrast)):
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"profile must include a finite {name} value.")

    @staticmethod
    def _add_noise_step(profile, config, steps, skipped) -> None:
        if profile.noise_level == "High":
            steps.append(
                RestorationStep(
                    "non_local_means_denoise",
                    "Noise reduction — Non-Local Means",
                    config.noise_high_parameters,
                    "High noise detected.",
                )
            )
        elif profile.noise_level == "Medium":
            steps.append(
                RestorationStep(
                    "bilateral_denoise",
                    "Noise reduction — Bilateral Filter",
                    config.noise_medium_parameters,
                    "Medium noise detected.",
                )
            )
        else:
            skipped.append("Noise reduction — noise level is Low.")

    @staticmethod
    def _add_color_step(profile, steps, skipped) -> None:
        if profile.color_cast != "None":
            steps.append(
                RestorationStep(
                    "white_balance",
                    "Color correction — White Balance",
                    {},
                    f"{profile.color_cast} color cast detected.",
                )
            )
        else:
            skipped.append("Color correction — no color cast detected.")

    @staticmethod
    def _add_tone_steps(profile, config, steps, skipped) -> None:
        if profile.brightness < 75:
            steps.append(
                RestorationStep(
                    "gamma",
                    "Tone correction — Gamma",
                    {"gamma": config.dark_gamma},
                    "Image brightness is low.",
                )
            )
        elif profile.brightness > 180:
            steps.append(
                RestorationStep(
                    "gamma",
                    "Tone correction — Gamma",
                    {"gamma": config.bright_gamma},
                    "Image brightness is high.",
                )
            )
        else:
            skipped.append("Tone correction — brightness is within the configured range.")

        if profile.contrast < config.contrast_threshold:
            steps.append(
                RestorationStep(
                    "clahe",
                    "Contrast enhancement — CLAHE",
                    {"clip_limit": config.clahe_clip_limit, "tile_grid_size": (8, 8)},
                    "Image contrast is below the selected strength threshold.",
                )
            )
        else:
            skipped.append("Contrast enhancement — contrast is already adequate.")

    @staticmethod
    def _add_deblur_step(profile, config, steps, skipped) -> None:
        amount = 0.0
        if profile.blur_level == "High":
            amount = config.high_blur_unsharp_amount
        elif profile.blur_level == "Medium":
            amount = config.medium_blur_unsharp_amount
        if amount > 0:
            steps.append(
                RestorationStep(
                    "unsharp",
                    "Detail recovery — Unsharp Mask",
                    {"radius": 1.0, "amount": amount, "threshold": 3},
                    f"{profile.blur_level} blur detected.",
                )
            )
        else:
            skipped.append("Detail recovery — blur level is below the configured threshold.")

    @staticmethod
    def _add_final_sharpen_step(profile, config, steps, skipped) -> None:
        if profile.noise_level == "High" and profile.blur_level == "High":
            skipped.append("Final sharpening — skipped to limit artifacts on heavily degraded detail.")
            return
        steps.append(
            RestorationStep(
                "sharpen",
                "Final refinement — Controlled Sharpening",
                {"amount": config.final_sharpen_amount},
                "Applied after restoration for controlled detail refinement.",
            )
        )

    @staticmethod
    def _add_style_step(style, config, steps) -> None:
        if style == "Vintage":
            steps.append(
                RestorationStep(
                    "vintage",
                    "Style — Vintage",
                    {"intensity": config.vintage_intensity},
                    "Applied after restoration and refinement.",
                )
            )
        elif style == "B&W":
            steps.append(
                RestorationStep(
                    "contrast",
                    "Style refinement — B&W Contrast",
                    {"factor": 1.05},
                    "Applied before monochrome conversion.",
                )
            )
            steps.append(
                RestorationStep(
                    "black_white",
                    "Style — B&W",
                    {},
                    "Applied after restoration and tonal refinement.",
                )
            )
        else:
            steps.append(
                RestorationStep(
                    "contrast",
                    "Style — Natural",
                    {"factor": 1.0},
                    "Natural styling preserves the restored tonal appearance.",
                )
            )
