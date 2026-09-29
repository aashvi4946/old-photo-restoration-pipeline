from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.image_viewer import ImageViewer
from ui.quick_restore_panel import QuickRestorePanel
from ui.repair_panel import RepairPanel
from ui.advanced_panel import AdvancedPanel
from ui.analysis_panel import AnalysisPanel
from ui.toolbar import AppToolbar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Old Photo Restorer")
        self.resize(1280, 800)

        self._setup_ui()

    def set_active_mode(self, mode):
        self.active_mode = mode

        self.quick_restore_button.setChecked(mode == "quick_restore")
        self.photo_repair_button.setChecked(mode == "photo_repair")
        self.advanced_button.setChecked(mode == "advanced")

        if mode == "quick_restore":
            self.mode_stack.setCurrentWidget(self.quick_restore_panel)

        elif mode == "photo_repair":
            self.mode_stack.setCurrentWidget(self.repair_panel)

        elif mode == "advanced":
            self.mode_stack.setCurrentWidget(self.advanced_panel)

    def _setup_ui(self):
        # ---------------------------------------------------------
        # Toolbar
        # ---------------------------------------------------------
        self.toolbar = AppToolbar(self)
        self.addToolBar(self.toolbar)

        # ---------------------------------------------------------
        # Menu bar
        # ---------------------------------------------------------
        file_menu = self.menuBar().addMenu("File")
        edit_menu = self.menuBar().addMenu("Edit")
        view_menu = self.menuBar().addMenu("View")
        help_menu = self.menuBar().addMenu("Help")

        file_menu.addAction(self.toolbar.open_action)
        file_menu.addAction(self.toolbar.save_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu.addAction(self.toolbar.undo_action)
        edit_menu.addAction(self.toolbar.redo_action)
        edit_menu.addAction(self.toolbar.reset_action)

        about_action = QAction("About", self)
        help_menu.addAction(about_action)

        # ---------------------------------------------------------
        # Central widget
        # ---------------------------------------------------------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------
        header = QFrame()
        header.setObjectName("header")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)

        title_label = QLabel("Old Photo Restorer")
        title_label.setObjectName("applicationTitle")

        header_layout.addWidget(title_label)
        header_layout.addStretch()

        main_layout.addWidget(header)

        # ---------------------------------------------------------
        # Main content area
        # ---------------------------------------------------------
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)

        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        # ---------------------------------------------------------
        # Left panel
        # ---------------------------------------------------------
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")

        left_layout = QVBoxLayout(left_panel)

        nav_title = QLabel("Modes")
        left_layout.addWidget(nav_title)

        self.quick_restore_button = QPushButton("Quick Restore")
        self.photo_repair_button = QPushButton("Photo Repair")
        self.advanced_button = QPushButton("Advanced")

        self.quick_restore_button.setCheckable(True)
        self.photo_repair_button.setCheckable(True)
        self.advanced_button.setCheckable(True)

        left_layout.addWidget(self.quick_restore_button)
        left_layout.addWidget(self.photo_repair_button)
        left_layout.addWidget(self.advanced_button)

        # ---------------------------------------------------------
        # Mode panels
        # ---------------------------------------------------------
        self.mode_stack = QStackedWidget()

        self.quick_restore_panel = QuickRestorePanel()
        self.repair_panel = RepairPanel()
        self.advanced_panel = AdvancedPanel()

        self.mode_stack.addWidget(self.quick_restore_panel)
        self.mode_stack.addWidget(self.repair_panel)
        self.mode_stack.addWidget(self.advanced_panel)

        left_layout.addWidget(self.mode_stack)
        left_layout.addStretch()

        # ---------------------------------------------------------
        # Central workspace
        # ---------------------------------------------------------
        center_panel = ImageViewer()

        # ---------------------------------------------------------
        # Right panel
        # ---------------------------------------------------------
        # right_panel = QFrame()
        # right_panel.setObjectName("rightPanel")

        # right_layout = QVBoxLayout(right_panel)

        # right_label = QLabel("Information")
        # right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # right_layout.addWidget(right_label)
        # right_layout.addStretch()

        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")

        right_layout = QVBoxLayout(right_panel)

        self.analysis_panel = AnalysisPanel()

        right_layout.addWidget(self.analysis_panel)

        # ---------------------------------------------------------
        # Add panels to content layout
        # ---------------------------------------------------------
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(center_panel, 3)
        content_layout.addWidget(right_panel, 1)

        main_layout.addWidget(content_widget, 1)

        # ---------------------------------------------------------
        # Button connections
        # ---------------------------------------------------------
        self.quick_restore_button.clicked.connect(
            lambda: self.set_active_mode("quick_restore")
        )

        self.photo_repair_button.clicked.connect(
            lambda: self.set_active_mode("photo_repair")
        )

        self.advanced_button.clicked.connect(
            lambda: self.set_active_mode("advanced")
        )

        # ---------------------------------------------------------
        # Status bar
        # ---------------------------------------------------------
        self.statusBar().showMessage("Ready")

        # Start with Quick Restore selected
        self.set_active_mode("quick_restore")