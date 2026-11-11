import numpy as np
import pytest

from core.pipeline import OPERATION_DEFINITIONS, ProcessingPipeline


def test_operation_browser_definitions_cover_prepare_restore_and_finish():
    categories = {definition.category for definition in OPERATION_DEFINITIONS}
    identifiers = {definition.identifier for definition in OPERATION_DEFINITIONS}

    assert categories == {"Prepare", "Restore", "Finish"}
    assert {"clahe", "non_local_means", "inpaint", "vintage", "adaptive_threshold"} <= identifiers


def test_pipeline_adds_default_parameters_updates_and_reorders_steps():
    pipeline = ProcessingPipeline()
    pipeline.add("brightness")
    pipeline.add("contrast")

    pipeline.update_parameters(0, {"value": 25})
    target = pipeline.move(1, -1)

    assert target == 0
    assert [step.operation for step in pipeline.steps] == ["contrast", "brightness"]
    assert pipeline.steps[1].parameters == {"value": 25}


def test_pipeline_removes_and_clears_steps():
    pipeline = ProcessingPipeline()
    pipeline.add("brightness")
    pipeline.add("gamma")

    removed = pipeline.remove(0)
    pipeline.clear()

    assert removed.operation == "brightness"
    assert pipeline.is_empty


def test_pipeline_executes_ordered_operations_without_mutating_input():
    image = np.full((8, 8), 100, dtype=np.uint8)
    original = image.copy()
    pipeline = ProcessingPipeline()
    pipeline.add("brightness")
    pipeline.update_parameters(0, {"value": 20})
    pipeline.add("gamma")
    pipeline.update_parameters(1, {"gamma": 2.0})

    result = pipeline.execute(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result[0, 0] > 120
    assert np.array_equal(image, original)


def test_pipeline_uses_psf_adapter_and_preserves_bgra_alpha():
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[6:10, 6:10] = (50, 100, 180, 123)
    image[:, :, 3] = np.arange(16, dtype=np.uint8)
    alpha = image[:, :, 3].copy()
    pipeline = ProcessingPipeline()
    pipeline.add("wiener")
    pipeline.update_parameters(0, {"psf_length": 3, "psf_angle": 0.0, "balance": 0.1})

    result = pipeline.execute(image)

    assert result.shape == image.shape
    assert np.array_equal(result[:, :, 3], alpha)


def test_pipeline_inpainting_uses_supplied_repair_mask():
    image = np.full((20, 20), 200, dtype=np.uint8)
    image[10, 10] = 0
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    pipeline = ProcessingPipeline()
    pipeline.add("inpaint")

    result = pipeline.execute(image, repair_mask=mask)

    assert result[10, 10] > image[10, 10]


def test_pipeline_adaptive_threshold_supports_color_images_and_alpha():
    image = np.full((16, 16, 4), (100, 130, 160, 77), dtype=np.uint8)
    image[:8, :, :3] = 30
    alpha = image[:, :, 3].copy()
    pipeline = ProcessingPipeline()
    pipeline.add("adaptive_threshold")

    result = pipeline.execute(image)

    assert result.shape == image.shape
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])
    assert np.array_equal(result[:, :, 3], alpha)


@pytest.mark.parametrize("operation", ["unknown", "", None])
def test_pipeline_rejects_unknown_operations(operation):
    with pytest.raises(ValueError):
        ProcessingPipeline().add(operation)


def test_pipeline_rejects_empty_execution_invalid_updates_and_bad_moves():
    pipeline = ProcessingPipeline()
    with pytest.raises(ValueError):
        pipeline.execute(np.zeros((3, 3), dtype=np.uint8))
    pipeline.add("brightness")
    with pytest.raises(ValueError):
        pipeline.update_parameters(0, {"wrong": 1})
    with pytest.raises(ValueError):
        pipeline.move(0, 1)
