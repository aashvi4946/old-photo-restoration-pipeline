import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.prepare_panel import PreparePanel


def test_prepare_panel_disables_actions_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = PreparePanel()

    assert not panel.histogram_button.isEnabled()
    assert not panel.clahe_button.isEnabled()
    assert not panel.white_balance_button.isEnabled()
    assert not panel.normalize_button.isEnabled()
    assert not panel.resize_button.isEnabled()

    panel.close()


def test_prepare_panel_enables_actions_when_image_dimensions_are_set():
    app = QApplication.instance() or QApplication([])
    panel = PreparePanel()

    panel.set_image_dimensions((800, 400))

    assert panel.histogram_button.isEnabled()
    assert panel.clahe_button.isEnabled()
    assert panel.white_balance_button.isEnabled()
    assert panel.normalize_button.isEnabled()
    assert panel.resize_button.isEnabled()

    panel.close()


def test_prepare_panel_emits_simple_operation_requests():
    app = QApplication.instance() or QApplication([])
    panel = PreparePanel()
    panel.set_image_dimensions((800, 400))
    requests = []
    panel.operation_requested.connect(lambda operation, parameters: requests.append((operation, parameters)))

    panel.histogram_button.click()
    panel.white_balance_button.click()
    panel.normalize_button.click()

    assert requests == [
        ("histogram", {}),
        ("white_balance", {}),
        ("normalize", {}),
    ]

    panel.close()


def test_resize_controls_maintain_aspect_ratio_by_default():
    app = QApplication.instance() or QApplication([])
    panel = PreparePanel()
    panel.set_image_dimensions((800, 400))

    panel.resize_width_input.setValue(400)

    assert panel.resize_height_input.value() == 200

    panel.close()


def test_clahe_parameters_expand_inline_and_emit_request():
    app = QApplication.instance() or QApplication([])
    panel = PreparePanel()
    panel.set_image_dimensions((800, 400))
    requests = []
    panel.operation_requested.connect(
        lambda operation, parameters: requests.append((operation, parameters))
    )

    panel.clahe_button.click()

    assert not panel.clahe_controls.isHidden()
    assert panel.resize_controls.isHidden()

    panel.clahe_clip_limit.setValue(3.0)
    panel._apply_clahe()

    assert requests == [("clahe", {"clip_limit": 3.0, "tile_grid_size": (8, 8)})]
    assert panel.clahe_controls.isHidden()

    panel.close()
