from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RepairPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 16, 0, 16)

        # Title
        layout.addWidget(QLabel("Photo Repair"))

        # Tool
        layout.addWidget(QLabel("Tool"))

        tool = QComboBox()
        tool.setMinimumHeight(42)
        tool.setMaxVisibleItems(2)
        tool.addItems([
            "Brush",
            "Eraser",
        ])

        layout.addWidget(tool)

        # Inpainting Method
        layout.addWidget(QLabel("Method"))

        method = QComboBox()
        method.setMinimumHeight(42)
        method.setMaxVisibleItems(2)
        method.addItems([
            "Telea",
            "Navier-Stokes",
        ])

        layout.addWidget(method)

        # Buttons
        clear_button = QPushButton("Clear")
        clear_button.setMinimumHeight(42)

        apply_button = QPushButton("Apply Repair")
        apply_button.setMinimumHeight(42)

        layout.addWidget(clear_button)
        layout.addWidget(apply_button)

        layout.addStretch()