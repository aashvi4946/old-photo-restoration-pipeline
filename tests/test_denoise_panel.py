import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from ui.denoise_panel import DenoisePanel


def test_denoise_panel_disables_actions_without_an_image():
    app = QApplication.instance() or QApplication([])
    panel = DenoisePanel()

    assert not panel.gaussian_button.isEnabled()
    assert not panel.median_button.isEnabled()
    assert not panel.bilateral_button.isEnabled()
    assert not panel.non_local_means_button.isEnabled()

    panel.close()


def test_denoise_panel_enables_actions_when_an_image_is_available():
    app = QApplication.instance() or QApplication([])
    panel = DenoisePanel()

    panel.set_image_available(True)

    assert panel.gaussian_button.isEnabled()
    assert panel.median_button.isEnabled()
    assert panel.bilateral_button.isEnabled()
    assert panel.non_local_means_button.isEnabled()

    panel.close()


def test_denoise_panel_expands_one_inline_control_section_at_a_time():
    app = QApplication.instance() or QApplication([])
    panel = DenoisePanel()
    panel.set_image_available(True)

    panel.gaussian_button.click()
    assert not panel.gaussian_controls.isHidden()

    panel.bilateral_button.click()
    assert panel.gaussian_controls.isHidden()
    assert not panel.bilateral_controls.isHidden()

    panel.close()


def test_denoise_panel_emits_gaussian_parameters():
    app = QApplication.instance() or QApplication([])
    panel = DenoisePanel()
    panel.set_image_available(True)
    requests = []
    panel.operation_requested.connect(
        lambda operation, parameters: requests.append((operation, parameters))
    )

    panel.gaussian_kernel.setCurrentText("7")
    panel.gaussian_sigma.setValue(2.0)
    panel._apply_gaussian()

    assert requests == [("gaussian", {"kernel_size": 7, "sigma": 2.0})]
    assert panel.gaussian_controls.isHidden()

    panel.close()
