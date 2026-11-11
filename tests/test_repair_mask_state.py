import numpy as np
import pytest

from core.repair_mask_state import RepairMaskState


def test_mask_state_creates_a_binary_mask_for_image_dimensions():
    state = RepairMaskState()
    state.set_image_shape(20, 30)

    assert state.mask.shape == (20, 30)
    assert state.mask.dtype == np.uint8
    assert not state.has_damage


def test_brush_stroke_is_undoable_and_redoable():
    state = RepairMaskState()
    state.set_image_shape(20, 20)
    state.begin_stroke()
    state.paint_line((2, 2), (10, 2), brush_size=3)

    assert state.end_stroke()
    painted = state.mask.copy()
    assert state.has_damage
    assert state.can_undo

    assert state.undo()
    assert not state.has_damage
    assert state.can_redo

    assert state.redo()
    assert np.array_equal(state.mask, painted)


def test_eraser_removes_existing_mask_area_as_an_undoable_stroke():
    state = RepairMaskState()
    state.set_image_shape(20, 20)
    state.begin_stroke()
    state.paint_line((2, 2), (10, 2), brush_size=5)
    state.end_stroke()
    before_erase = state.mask.copy()

    state.begin_stroke()
    state.paint_line((2, 2), (10, 2), brush_size=5, erase=True)
    state.end_stroke()

    assert not state.has_damage
    assert state.undo()
    assert np.array_equal(state.mask, before_erase)


def test_clear_mask_is_undoable():
    state = RepairMaskState()
    state.set_image_shape(20, 20)
    state.begin_stroke()
    state.paint_line((2, 2), (10, 2), brush_size=3)
    state.end_stroke()
    painted = state.mask.copy()

    assert state.clear_all()
    assert not state.has_damage
    assert state.undo()
    assert np.array_equal(state.mask, painted)


def test_mask_state_rejects_invalid_brush_size():
    state = RepairMaskState()
    state.set_image_shape(20, 20)

    with pytest.raises(ValueError):
        state.paint_line((1, 1), (2, 2), brush_size=0)
