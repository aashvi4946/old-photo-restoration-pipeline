import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.quick_restore_panel import QuickRestorePanel


def test_quick_restore_panel_is_disabled_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = QuickRestorePanel()

    assert not panel.strength.isEnabled()
    assert not panel.style.isEnabled()
    assert not panel.restore_button.isEnabled()

    panel.close()


def test_quick_restore_panel_emits_selected_strength_and_style_and_shows_report():
    app = QApplication.instance() or QApplication([])
    panel = QuickRestorePanel()
    panel.set_image_available(True)
    requests = []
    panel.restore_requested.connect(lambda strength, style: requests.append((strength, style)))

    panel.strength.setCurrentText("Strong")
    panel.style.setCurrentText("B&W")
    panel.restore_button.click()
    panel.update_report("RESTORATION REPORT\n✓ CLAHE")

    assert requests == [("Strong", "B&W")]
    assert panel.report_label.text() == "RESTORATION REPORT\n✓ CLAHE"
    panel.close()
