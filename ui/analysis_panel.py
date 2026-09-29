from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from models.image_profile import ImageProfile


class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        title = QLabel("IMAGE ANALYSIS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.metadata_label = QLabel("No image loaded.")
        self.blur_label = QLabel()
        self.noise_label = QLabel()
        self.brightness_label = QLabel()
        self.contrast_label = QLabel()
        self.color_cast_label = QLabel()

        layout.addWidget(title)
        for label in (
            self.metadata_label,
            self.blur_label,
            self.noise_label,
            self.brightness_label,
            self.contrast_label,
            self.color_cast_label,
        ):
            label.setWordWrap(True)
            layout.addWidget(label)

        # layout.addWidget(title)
        layout.addStretch()

        self.clear()

    def update_profile(self, profile: ImageProfile) -> None:
        """Display an ImageProfile."""

        self.metadata_label.setText(
            f"Size: {profile.width} × {profile.height}\n"
            f"Channels: {profile.channels}\n"
            f"Grayscale: {'Yes' if profile.is_grayscale else 'No'}"
        )

        self.blur_label.setText(
            f"Blur: {profile.blur_level} "
            f"({profile.blur_score:.2f})"
        )

        self.noise_label.setText(
            f"Noise: {profile.noise_level} "
            f"({profile.noise_score:.2f})"
        )

        self.brightness_label.setText(
            f"Brightness: {profile.brightness:.2f}"
        )

        self.contrast_label.setText(
            f"Contrast: {profile.contrast:.2f}"
        )

        self.color_cast_label.setText(
            f"Color Cast: {profile.color_cast}"
        )

    def clear(self) -> None:
        """Clear analysis information."""

        self.metadata_label.setText("No image loaded.")
        self.blur_label.setText("Blur: —")
        self.noise_label.setText("Noise: —")
        self.brightness_label.setText("Brightness: —")
        self.contrast_label.setText("Contrast: —")
        self.color_cast_label.setText("Color Cast: —")
