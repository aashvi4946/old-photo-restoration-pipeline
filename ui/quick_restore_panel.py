from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QuickRestorePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 16, 0, 16)

        # Title
        layout.addWidget(QLabel("Quick Restore"))

        # Strength
        layout.addWidget(QLabel("Strength"))

        strength = QComboBox()
        strength.setMinimumHeight(42)
        strength.setMaxVisibleItems(3)
        strength.addItems([
            "Light",
            "Medium",
            "Strong",
        ])

        layout.addWidget(strength)

        # Style
        layout.addWidget(QLabel("Style"))

        style = QComboBox()
        style.setMinimumHeight(42)
        style.setMaxVisibleItems(3)
        style.addItems([
            "Natural",
            "Vintage",
            "B&W",
        ])

        layout.addWidget(style)

        # Restore button
        restore_button = QPushButton("Restore")
        restore_button.setMinimumHeight(42)

        layout.addWidget(restore_button)

        layout.addStretch()