import numpy as np
import pytest

from core.image_state import ImageState


def test_initial_state():
    state = ImageState()

    assert state.original_image is None
    assert state.current_image is None
    assert state.has_image is False
    assert state.can_undo is False
    assert state.can_redo is False


def test_set_image_creates_independent_original_and_current():
    image = np.zeros((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(image)

    assert state.has_image is True
    assert np.array_equal(state.original_image, image)
    assert np.array_equal(state.current_image, image)

    assert state.original_image is not state.current_image


def test_set_image_starts_new_history():
    image = np.zeros((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(image)

    changed = np.ones((10, 10), dtype=np.uint8)
    state.apply_change(changed)

    assert state.can_undo is True

    second_image = np.full((10, 10), 255, dtype=np.uint8)
    state.set_image(second_image)

    assert state.can_undo is False
    assert state.can_redo is False
    assert np.array_equal(state.current_image, second_image)


def test_apply_change_creates_undo_point():
    original = np.zeros((10, 10), dtype=np.uint8)
    changed = np.ones((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(original)
    state.apply_change(changed)

    assert np.array_equal(state.current_image, changed)
    assert state.can_undo is True
    assert state.can_redo is False


def test_undo_restores_previous_state():
    original = np.zeros((10, 10), dtype=np.uint8)
    changed = np.ones((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(original)
    state.apply_change(changed)

    assert state.undo() is True
    assert np.array_equal(state.current_image, original)
    assert state.can_redo is True


def test_redo_restores_undone_state():
    original = np.zeros((10, 10), dtype=np.uint8)
    changed = np.ones((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(original)
    state.apply_change(changed)

    state.undo()

    assert state.redo() is True
    assert np.array_equal(state.current_image, changed)


def test_new_change_clears_redo_history():
    original = np.zeros((10, 10), dtype=np.uint8)
    first_change = np.ones((10, 10), dtype=np.uint8)
    second_change = np.full((10, 10), 2, dtype=np.uint8)

    state = ImageState()
    state.set_image(original)
    state.apply_change(first_change)

    state.undo()
    assert state.can_redo is True

    state.apply_change(second_change)

    assert state.can_redo is False
    assert np.array_equal(state.current_image, second_change)


def test_multiple_undo_and_redo():
    original = np.zeros((10, 10), dtype=np.uint8)
    first = np.ones((10, 10), dtype=np.uint8)
    second = np.full((10, 10), 2, dtype=np.uint8)

    state = ImageState()
    state.set_image(original)
    state.apply_change(first)
    state.apply_change(second)

    assert state.undo() is True
    assert np.array_equal(state.current_image, first)

    assert state.undo() is True
    assert np.array_equal(state.current_image, original)

    assert state.undo() is False

    assert state.redo() is True
    assert np.array_equal(state.current_image, first)

    assert state.redo() is True
    assert np.array_equal(state.current_image, second)

    assert state.redo() is False


def test_original_is_never_modified():
    original = np.zeros((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(original)

    changed = np.full((10, 10), 255, dtype=np.uint8)
    state.apply_change(changed)

    state.undo()
    state.redo()

    assert np.all(state.original_image == 0)


def test_reset_restores_original():
    original = np.zeros((10, 10), dtype=np.uint8)
    changed = np.full((10, 10), 255, dtype=np.uint8)

    state = ImageState()
    state.set_image(original)
    state.apply_change(changed)

    assert state.reset() is True
    assert np.array_equal(state.current_image, original)


def test_reset_without_image():
    state = ImageState()

    assert state.reset() is False


def test_clear_removes_all_state():
    image = np.zeros((10, 10), dtype=np.uint8)

    state = ImageState()
    state.set_image(image)
    state.clear()

    assert state.original_image is None
    assert state.current_image is None
    assert state.has_image is False
    assert state.can_undo is False
    assert state.can_redo is False


def test_invalid_image_rejected():
    state = ImageState()

    with pytest.raises(ValueError):
        state.set_image(np.array([]))


def test_invalid_dimensions_rejected():
    state = ImageState()

    image = np.zeros((10, 10, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        state.set_image(image)


def test_apply_change_without_loaded_image():
    state = ImageState()
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        state.apply_change(image)


def test_load_image_creates_new_state(tmp_path):
    import cv2

    image = np.full((10, 10), 128, dtype=np.uint8)
    path = tmp_path / "test.png"

    assert cv2.imwrite(str(path), image)

    state = ImageState()
    state.load_image(path)

    assert state.has_image is True
    assert np.array_equal(state.original_image, image)
    assert np.array_equal(state.current_image, image)
    assert state.can_undo is False
    assert state.can_redo is False


def test_loading_new_image_clears_old_history(tmp_path):
    import cv2

    first = np.zeros((10, 10), dtype=np.uint8)
    first_path = tmp_path / "first.png"

    second = np.full((10, 10), 255, dtype=np.uint8)
    second_path = tmp_path / "second.png"

    assert cv2.imwrite(str(first_path), first)
    assert cv2.imwrite(str(second_path), second)

    state = ImageState()
    state.load_image(first_path)

    changed = np.full((10, 10), 100, dtype=np.uint8)
    state.apply_change(changed)

    assert state.can_undo is True

    state.load_image(second_path)

    assert np.array_equal(state.original_image, second)
    assert np.array_equal(state.current_image, second)
    assert state.can_undo is False
    assert state.can_redo is False


def test_failed_load_preserves_existing_state(tmp_path):
    import cv2

    original = np.full((10, 10), 50, dtype=np.uint8)
    valid_path = tmp_path / "valid.png"

    assert cv2.imwrite(str(valid_path), original)

    state = ImageState()
    state.load_image(valid_path)

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not a valid image")

    with pytest.raises(ValueError):
        state.load_image(invalid_path)

    assert np.array_equal(state.original_image, original)
    assert np.array_equal(state.current_image, original)
    assert state.can_undo is False
    assert state.can_redo is False
