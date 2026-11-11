import numpy as np
import pytest

from core.restoration_engine import RESTORATION_STRENGTHS, RestorationEngine
from models.image_profile import ImageProfile


def _profile(**overrides) -> ImageProfile:
    values = {
        "noise_level": "Low",
        "blur_level": "Low",
        "color_cast": "None",
        "brightness": 128.0,
        "contrast": 70.0,
    }
    values.update(overrides)
    return ImageProfile(**values)


def test_quick_restore_builds_an_explainable_ordered_plan_for_detected_issues():
    engine = RestorationEngine()
    profile = _profile(
        noise_level="High",
        blur_level="High",
        color_cast="Red",
        brightness=50.0,
        contrast=20.0,
    )

    plan = engine.build_plan(profile, strength="Strong", style="Vintage")

    assert [step.operation for step in plan.steps] == [
        "non_local_means_denoise",
        "white_balance",
        "gamma",
        "clahe",
        "unsharp",
        "vintage",
    ]
    assert plan.steps[0].parameters == RESTORATION_STRENGTHS["Strong"].noise_high_parameters
    assert plan.steps[3].parameters["clip_limit"] == 3.0
    assert "RESTORATION REPORT" in plan.report
    assert "High noise detected" in plan.report
    assert "Skipped:" in plan.report


def test_strength_configuration_changes_the_selected_operation_parameters():
    engine = RestorationEngine()
    profile = _profile(noise_level="Medium", contrast=30.0)

    light = engine.build_plan(profile, strength="Light")
    strong = engine.build_plan(profile, strength="Strong")

    light_noise = next(step for step in light.steps if step.operation == "bilateral_denoise")
    strong_noise = next(step for step in strong.steps if step.operation == "bilateral_denoise")
    light_clahe = next(step for step in light.steps if step.operation == "clahe")
    strong_clahe = next(step for step in strong.steps if step.operation == "clahe")
    assert light_noise.parameters["diameter"] < strong_noise.parameters["diameter"]
    assert light_clahe.parameters["clip_limit"] < strong_clahe.parameters["clip_limit"]


def test_medium_blur_is_only_deblurred_at_medium_or_strong_strength():
    engine = RestorationEngine()
    profile = _profile(blur_level="Medium")

    light = engine.build_plan(profile, strength="Light")
    medium = engine.build_plan(profile, strength="Medium")

    assert "unsharp" not in [step.operation for step in light.steps]
    assert "unsharp" in [step.operation for step in medium.steps]


def test_style_is_applied_after_restoration_and_bw_includes_tonal_refinement():
    engine = RestorationEngine()
    profile = _profile(noise_level="Medium", color_cast="Blue")

    vintage = engine.build_plan(profile, style="Vintage")
    black_white = engine.build_plan(profile, style="B&W")

    assert vintage.steps[-1].operation == "vintage"
    assert [step.operation for step in black_white.steps[-2:]] == [
        "contrast",
        "black_white",
    ]


def test_engine_executes_plan_without_mutating_input_or_alpha_channel():
    engine = RestorationEngine()
    image = np.full((32, 32, 4), (70, 90, 140, 123), dtype=np.uint8)
    original = image.copy()
    plan = engine.build_plan(
        _profile(color_cast="Red", contrast=20.0),
        strength="Medium",
        style="Vintage",
    )

    result = engine.execute(image, plan)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 3], original[:, :, 3])
    assert np.array_equal(image, original)


@pytest.mark.parametrize("strength", ["weak", "", None])
def test_engine_rejects_invalid_strength(strength):
    with pytest.raises(ValueError):
        RestorationEngine().build_plan(_profile(), strength=strength)


@pytest.mark.parametrize("style", ["Sepia", "", None])
def test_engine_rejects_invalid_style(style):
    with pytest.raises(ValueError):
        RestorationEngine().build_plan(_profile(), style=style)


def test_engine_rejects_incomplete_profile_and_non_plan_execution():
    engine = RestorationEngine()
    with pytest.raises(ValueError):
        engine.build_plan(ImageProfile())
    with pytest.raises(TypeError):
        engine.execute(np.zeros((3, 3), dtype=np.uint8), object())
