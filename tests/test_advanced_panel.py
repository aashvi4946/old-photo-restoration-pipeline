import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.advanced_panel import AdvancedPanel


def test_advanced_panel_disables_pipeline_controls_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = AdvancedPanel()

    assert not panel.add_button.isEnabled()
    assert not panel.apply_button.isEnabled()

    panel.close()


def test_advanced_panel_builds_reorders_and_removes_pipeline_steps():
    app = QApplication.instance() or QApplication([])
    panel = AdvancedPanel()
    panel.set_image_available(True)

    panel.operation_browser.setCurrentIndex(panel.operation_browser.findData("brightness"))
    panel.add_button.click()
    panel.operation_browser.setCurrentIndex(panel.operation_browser.findData("contrast"))
    panel.add_button.click()
    panel.move_up_button.click()

    assert [step.operation for step in panel.pipeline.steps] == ["contrast", "brightness"]
    assert panel.pipeline_list.count() == 2

    panel.remove_button.click()
    assert [step.operation for step in panel.pipeline.steps] == ["brightness"]

    panel.clear_button.click()
    assert panel.pipeline.is_empty
    panel.close()


def test_advanced_panel_updates_selected_advanced_parameters_and_emits_pipeline():
    app = QApplication.instance() or QApplication([])
    panel = AdvancedPanel()
    panel.set_image_available(True)
    requested = []
    panel.pipeline_requested.connect(requested.append)

    panel.operation_browser.setCurrentIndex(panel.operation_browser.findData("clahe"))
    panel.add_button.click()
    panel._parameter_widgets["clip_limit"].setValue(3.0)
    panel._parameter_widgets["tile_width"].setValue(12)
    panel.update_parameters_button.click()
    panel.apply_button.click()

    assert panel.pipeline.steps[0].parameters == {
        "clip_limit": 3.0,
        "tile_width": 12,
        "tile_height": 8,
    }
    assert requested == [panel.pipeline]
    panel.close()
