from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)


class AdvancedPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 16, 0, 16)

        # Title
        layout.addWidget(QLabel("Advanced"))

        # Processing sections
        layout.addWidget(QLabel("Preprocessing"))
        layout.addWidget(QLabel("Restoration"))
        layout.addWidget(QLabel("Post-processing"))
        layout.addWidget(QLabel("Parameters"))
        layout.addWidget(QLabel("Pipeline"))

        layout.addStretch()