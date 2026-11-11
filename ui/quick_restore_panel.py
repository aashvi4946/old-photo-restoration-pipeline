from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QuickRestorePanel(QWidget):
    """Simple controls for Phase 9's deterministic restoration workflow."""

    restore_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 16, 0, 16)

        layout.addWidget(QLabel("QUICK RESTORE"))

        # Strength
        layout.addWidget(QLabel("Strength"))

        self.strength = QComboBox()
        self.strength.setMinimumHeight(36)
        self.strength.setMaxVisibleItems(3)
        self.strength.addItems([
            "Light",
            "Medium",
            "Strong",
        ])

        layout.addWidget(self.strength)

        # Style
        layout.addWidget(QLabel("Style"))

        self.style = QComboBox()
        self.style.setMinimumHeight(36)
        self.style.setMaxVisibleItems(3)
        self.style.addItems([
            "Natural",
            "Vintage",
            "B&W",
        ])

        layout.addWidget(self.style)

        # Restore button
        self.restore_button = QPushButton("Restore")
        self.restore_button.setMinimumHeight(36)

        layout.addWidget(self.restore_button)

        self.report_label = QLabel("Load an image to create a restoration plan.")
        self.report_label.setWordWrap(True)
        layout.addWidget(self.report_label)

        layout.addStretch()

        self.restore_button.clicked.connect(self._request_restore)
        self.set_image_available(False)

    def set_image_available(self, available: bool) -> None:
        """Enable Quick Restore only after automatic image analysis is available."""

        self.strength.setEnabled(available)
        self.style.setEnabled(available)
        self.restore_button.setEnabled(available)
        if not available:
            self.clear_report()

    def update_report(self, report: str) -> None:
        """Display the explainable report returned by the restoration engine."""

        self.report_label.setText(report)

    def clear_report(self) -> None:
        """Clear an old report when the loaded image changes."""

        self.report_label.setText("Load an image to create a restoration plan.")

    def _request_restore(self) -> None:
        self.restore_requested.emit(
            self.strength.currentText(),
            self.style.currentText(),
        )
