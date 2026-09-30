import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def test_initial_action_states():
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    assert window.toolbar.open_action.isEnabled()
    assert not window.toolbar.save_action.isEnabled()
    assert not window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()
    assert not window.toolbar.reset_action.isEnabled()

    window.close()
    app.quit()


def test_action_states_after_loading_image(tmp_path):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    image = np.full((100, 100), 128, dtype=np.uint8)
    path = tmp_path / "test.png"
    assert cv2.imwrite(str(path), image)

    window.image_state.load_image(path)
    window._refresh_viewer()
    window._update_action_states()

    assert window.image_state.has_image
    assert not window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()
    assert window.toolbar.reset_action.isEnabled()
    assert not window.image_viewer.image_label.pixmap().isNull()

    window.close()
    app.quit()


def test_action_states_follow_undo_redo(tmp_path):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    image = np.zeros((100, 100), dtype=np.uint8)
    path = tmp_path / "test.png"
    assert cv2.imwrite(str(path), image)

    window.image_state.load_image(path)
    window._refresh_viewer()
    window._update_action_states()

    changed = np.full((100, 100), 255, dtype=np.uint8)
    window.image_state.apply_change(changed)
    window._refresh_viewer()
    window._update_action_states()

    assert window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()

    window.undo()

    assert not window.toolbar.undo_action.isEnabled()
    assert window.toolbar.redo_action.isEnabled()

    window.redo()

    assert window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()

    window.close()
    app.quit()


def test_reset_synchronizes_viewer_and_actions(tmp_path):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    image = np.full((100, 100), 50, dtype=np.uint8)
    path = tmp_path / "test.png"
    assert cv2.imwrite(str(path), image)

    window.image_state.load_image(path)
    window._refresh_viewer()
    window._update_action_states()

    changed = np.full((100, 100), 200, dtype=np.uint8)
    window.image_state.apply_change(changed)
    window._refresh_viewer()
    window._update_action_states()

    window.reset()

    assert np.array_equal(
        window.image_state.current_image,
        window.image_state.original_image,
    )

    # Reset creates an undo point for the pre-reset state.
    assert window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()
    assert window.toolbar.reset_action.isEnabled()
    assert not window.image_viewer.image_label.pixmap().isNull()

    window.close()
    app.quit()


def test_opening_new_image_replaces_previous_ui_state(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    first = np.zeros((100, 100), dtype=np.uint8)
    first_path = tmp_path / "first.png"
    assert cv2.imwrite(str(first_path), first)

    second = np.full((100, 100), 255, dtype=np.uint8)
    second_path = tmp_path / "second.png"
    assert cv2.imwrite(str(second_path), second)

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(first_path), "Image Files"),
    )

    window.open_image()

    changed = np.full((100, 100), 100, dtype=np.uint8)
    window.image_state.apply_change(changed)
    window._refresh_viewer()
    window._update_action_states()

    assert window.toolbar.undo_action.isEnabled()

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(second_path), "Image Files"),
    )

    window.open_image()

    assert np.array_equal(
        window.image_state.original_image,
        second,
    )
    assert np.array_equal(
        window.image_state.current_image,
        second,
    )

    assert not window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()
    assert window.toolbar.reset_action.isEnabled()

    window.close()
    app.quit()


def test_failed_open_preserves_existing_ui_state(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    valid_image = np.full((100, 100), 50, dtype=np.uint8)
    valid_path = tmp_path / "valid.png"
    assert cv2.imwrite(str(valid_path), valid_image)

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not a valid image")

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(valid_path), "Image Files"),
    )

    window.open_image()

    changed = np.full((100, 100), 150, dtype=np.uint8)
    window.image_state.apply_change(changed)
    window._refresh_viewer()
    window._update_action_states()

    original_before = window.image_state.original_image.copy()
    current_before = window.image_state.current_image.copy()

    monkeypatch.setattr(
        "ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(invalid_path), "Image Files"),
    )

    messages = []

    monkeypatch.setattr(
        "ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: messages.append(args),
    )

    window.open_image()

    assert len(messages) == 1

    assert np.array_equal(
        window.image_state.original_image,
        original_before,
    )
    assert np.array_equal(
        window.image_state.current_image,
        current_before,
    )

    assert window.toolbar.undo_action.isEnabled()
    assert not window.toolbar.redo_action.isEnabled()
    assert window.toolbar.reset_action.isEnabled()
    assert not window.image_viewer.image_label.pixmap().isNull()

    window.close()
    app.quit()


def test_viewer_is_not_authoritative_state(tmp_path):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    image = np.full((100, 100), 80, dtype=np.uint8)
    path = tmp_path / "test.png"
    assert cv2.imwrite(str(path), image)

    window.image_state.load_image(path)
    window._refresh_viewer()
    window._update_action_states()

    state_before = window.image_state.current_image.copy()

    window.image_viewer.clear()

    assert np.array_equal(
        window.image_state.current_image,
        state_before,
    )
    assert window.image_state.has_image is True
    assert window.toolbar.reset_action.isEnabled()

    window.close()
    app.quit()


def test_analysis_is_updated_when_image_is_loaded(tmp_path):
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    image = np.zeros((100, 100), dtype=np.uint8)
    path = tmp_path / "analysis.png"
    assert cv2.imwrite(str(path), image)

    window.image_state.load_image(path)
    window._refresh_analysis()

    assert window.image_profile is not None
    assert window.image_profile.width == 100
    assert window.image_profile.height == 100
    assert window.image_profile.channels == 1
    assert window.image_profile.is_grayscale is True

    assert "100" in window.analysis_panel.metadata_label.text()
    assert "Blur:" in window.analysis_panel.blur_label.text()
    assert "Noise:" in window.analysis_panel.noise_label.text()

    window.close()
    app.quit()


def test_analysis_refreshes_after_undo_redo_and_reset():
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    original = np.zeros((100, 100), dtype=np.uint8)
    window.image_state.set_image(original)
    window._refresh_analysis()

    assert window.image_profile.brightness == 0.0

    changed = np.full((100, 100), 200, dtype=np.uint8)
    window.image_state.apply_change(changed)

    window._refresh_analysis()

    assert window.image_profile.brightness == 200.0

    window.undo()
    window._refresh_analysis()

    assert window.image_profile.brightness == 0.0

    window.redo()
    assert window.image_profile.brightness == 200.0

    window.reset()
    assert window.image_profile.brightness == 0.0

    window.close()
    app.quit()


def test_preprocessing_operation_commits_to_state_and_refreshes_ui():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = np.tile(np.arange(100, 111, dtype=np.uint8), (20, 1))
    window.image_state.set_image(image)
    window._refresh_analysis()
    window._refresh_viewer()
    window._update_action_states()

    window._apply_preprocessing("histogram", {})

    assert not np.array_equal(window.image_state.current_image, image)
    assert np.array_equal(window.image_state.original_image, image)
    assert window.image_state.can_undo
    assert window.toolbar.undo_action.isEnabled()
    assert window.image_profile.brightness != float(image.mean())

    window.undo()
    assert np.array_equal(window.image_state.current_image, image)
    window.redo()
    assert not np.array_equal(window.image_state.current_image, image)

    window.close()
    app.quit()


def test_failed_preprocessing_preserves_current_image_and_history(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = np.full((20, 20), 100, dtype=np.uint8)
    window.image_state.set_image(image)
    current_before = window.image_state.current_image.copy()
    warnings = []
    monkeypatch.setattr(
        "ui.main_window.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window._apply_preprocessing("clahe", {"clip_limit": 0})

    assert np.array_equal(window.image_state.current_image, current_before)
    assert not window.image_state.can_undo
    assert len(warnings) == 1

    window.close()
    app.quit()


def test_resize_preprocessing_updates_analysis_and_prepare_dimensions():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = np.full((100, 80, 4), (50, 80, 120, 200), dtype=np.uint8)
    window.image_state.set_image(image)
    window._refresh_analysis()
    window._refresh_viewer()
    window._update_action_states()

    window._apply_preprocessing("resize", {"width": 40, "height": 50})

    assert window.image_state.current_image.shape == (50, 40, 4)
    assert window.image_profile.width == 40
    assert window.image_profile.height == 50
    assert window.prepare_panel._image_dimensions == (40, 50)
    assert np.all(window.image_state.current_image[:, :, 3] == 200)

    window.close()
    app.quit()


def test_denoising_operation_commits_to_state_and_refreshes_ui():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 255
    window.image_state.set_image(image)
    window._refresh_analysis()
    window._refresh_viewer()
    window._update_action_states()

    window._apply_denoising("median", {"kernel_size": 3})

    assert window.image_state.current_image[4, 4] == 0
    assert np.array_equal(window.image_state.original_image, image)
    assert window.image_state.can_undo
    assert window.toolbar.undo_action.isEnabled()

    window.undo()
    assert np.array_equal(window.image_state.current_image, image)

    window.close()
    app.quit()


def test_failed_denoising_preserves_current_image_and_history(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = np.zeros((9, 9), dtype=np.uint8)
    window.image_state.set_image(image)
    current_before = window.image_state.current_image.copy()
    warnings = []
    monkeypatch.setattr(
        "ui.main_window.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window._apply_denoising("gaussian", {"kernel_size": 2})

    assert np.array_equal(window.image_state.current_image, current_before)
    assert not window.image_state.can_undo
    assert len(warnings) == 1

    window.close()
    app.quit()
