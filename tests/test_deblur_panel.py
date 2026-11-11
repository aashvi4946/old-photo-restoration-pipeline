import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.deblur_panel import DeblurPanel


def test_deblur_panel_disables_actions_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = DeblurPanel()

    assert not panel.unsharp_button.isEnabled()
    assert not panel.wiener_button.isEnabled()
    assert not panel.lucy_richardson_button.isEnabled()

    panel.close()


def test_deblur_panel_enables_actions_when_an_image_is_available():
    app = QApplication.instance() or QApplication([])
    panel = DeblurPanel()

    panel.set_image_available(True)

    assert panel.unsharp_button.isEnabled()
    assert panel.wiener_button.isEnabled()
    assert panel.lucy_richardson_button.isEnabled()

    panel.close()


def test_deblur_panel_expands_one_inline_control_section_at_a_time():
    app = QApplication.instance() or QApplication([])
    panel = DeblurPanel()
    panel.set_image_available(True)

    panel.unsharp_button.click()
    assert not panel.unsharp_controls.isHidden()

    panel.wiener_button.click()
    assert panel.unsharp_controls.isHidden()
    assert not panel.wiener_controls.isHidden()

    panel.close()


def test_deblur_panel_emits_wiener_parameters():
    app = QApplication.instance() or QApplication([])
    panel = DeblurPanel()
    panel.set_image_available(True)
    requests = []
    panel.operation_requested.connect(
        lambda operation, parameters: requests.append((operation, parameters))
    )

    panel.wiener_psf_length.setCurrentText("15")
    panel.wiener_psf_angle.setValue(45.0)
    panel.wiener_balance.setValue(0.1)
    panel._apply_wiener()

    assert requests == [
        (
            "wiener",
            {"psf_length": 15, "psf_angle": 45.0, "balance": 0.1},
        )
    ]
    assert panel.wiener_controls.isHidden()

    panel.close()
