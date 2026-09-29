from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Old Photo Restorer")
        self.resize(800, 500)

        label = QLabel(
            "Old Photo Restorer\n\n"
            "Phase 0\n"
            "Project setup successful."
        )

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(label)
