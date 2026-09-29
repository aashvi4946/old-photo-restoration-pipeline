from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        title = QLabel("IMAGE ANALYSIS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        message = QLabel(
            "No image loaded.\n\n"
            "Analysis will appear here once an image is loaded."
        )
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(message)
        layout.addStretch()
