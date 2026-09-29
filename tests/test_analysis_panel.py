import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from models.image_profile import ImageProfile
from ui.analysis_panel import AnalysisPanel


def test_analysis_panel_can_display_profile():
    app = QApplication.instance() or QApplication([])

    panel = AnalysisPanel()

    profile = ImageProfile(
        width=100,
        height=200,
        channels=3,
        is_grayscale=False,
        blur_score=150.0,
        blur_level="Medium",
        noise_score=8.0,
        noise_level="Medium",
        brightness=120.0,
        contrast=40.0,
        color_cast="None",
    )

    panel.update_profile(profile)

    assert "100" in panel.metadata_label.text()
    assert "Medium" in panel.blur_label.text()
    assert "120.00" in panel.brightness_label.text()

    panel.close()
    app.quit()
