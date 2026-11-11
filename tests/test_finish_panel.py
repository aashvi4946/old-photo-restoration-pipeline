import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.finish_panel import FinishPanel


def test_finish_panel_disables_actions_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = FinishPanel()

    assert not panel.sharpen_button.isEnabled()
    assert not panel.black_white_button.isEnabled()
    assert not panel.vintage_button.isEnabled()

    panel.close()


def test_finish_panel_expands_one_inline_control_section_at_a_time():
    app = QApplication.instance() or QApplication([])
    panel = FinishPanel()
    panel.set_image_available(True)

    panel.sharpen_button.click()
    assert not panel.sharpen_controls.isHidden()

    panel.gamma_button.click()
    assert panel.sharpen_controls.isHidden()
    assert not panel.gamma_controls.isHidden()

    panel.close()


def test_finish_panel_emits_operations_and_parameters():
    app = QApplication.instance() or QApplication([])
    panel = FinishPanel()
    panel.set_image_available(True)
    requests = []
    panel.operation_requested.connect(
        lambda operation, parameters: requests.append((operation, parameters))
    )

    panel.brightness_value.setValue(-25)
    panel._apply_brightness()
    panel.red_gain.setValue(1.5)
    panel._apply_color_correction()
    panel.black_white_button.click()

    assert requests == [
        ("brightness", {"value": -25}),
        (
            "color_correction",
            {"blue_gain": 1.0, "green_gain": 1.0, "red_gain": 1.5},
        ),
        ("black_white", {}),
    ]
    panel.close()
