import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.repair_panel import RepairPanel


def test_repair_panel_disables_controls_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = RepairPanel()

    assert not panel.brush_button.isEnabled()
    assert not panel.apply_button.isEnabled()

    panel.close()


def test_repair_panel_emits_selected_inpainting_method_and_radius():
    app = QApplication.instance() or QApplication([])
    panel = RepairPanel()
    panel.set_repair_available(True, True, False, False)
    requests = []
    panel.inpaint_requested.connect(lambda method, radius: requests.append((method, radius)))
    panel.method.setCurrentIndex(1)
    panel.radius.setValue(5.0)

    panel.apply_button.click()

    assert requests == [("navier_stokes", 5.0)]
    panel.close()
