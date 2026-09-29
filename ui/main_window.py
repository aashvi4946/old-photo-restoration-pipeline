import cv2

from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QWidget,
    QHBoxLayout,
)
from core.image_state import ImageState
from processing.analysis.analyze import analyze_image
from processing.preprocessing.clahe import apply_clahe
from processing.preprocessing.histogram import equalize_histogram
from processing.preprocessing.normalization import normalize_image
from processing.preprocessing.resize import resize_image
from processing.preprocessing.white_balance import auto_white_balance
from models.image_profile import ImageProfile
from ui.toolbar import AppToolbar
from ui.image_viewer import ImageViewer
from ui.analysis_panel import AnalysisPanel
from ui.prepare_panel import PreparePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Old Photo Restorer")
        self.resize(1200, 800)

        self.image_state = ImageState()
        self.image_profile: ImageProfile | None = None

        self._setup_ui()
        self._connect_actions()
        self._update_action_states()

    def _setup_ui(self):
        self.toolbar = AppToolbar(self)
        self.addToolBar(self.toolbar)

        self.image_viewer = ImageViewer()
        self.analysis_panel = AnalysisPanel()
        self.prepare_panel = PreparePanel()

        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.prepare_panel, 0)
        layout.addWidget(self.image_viewer, 1)
        layout.addWidget(self.analysis_panel, 0)

        self.setCentralWidget(central_widget)
        
    def _connect_actions(self):
        self.toolbar.open_action.triggered.connect(self.open_image)
        self.toolbar.undo_action.triggered.connect(self.undo)
        self.toolbar.redo_action.triggered.connect(self.redo)
        self.toolbar.reset_action.triggered.connect(self.reset)
        self.prepare_panel.operation_requested.connect(self._apply_preprocessing)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            (
                "Image Files (*.jpg *.jpeg *.png *.tif *.tiff "
                "*.bmp *.webp)"
            ),
        )

        if not path:
            return

        try:
            self.image_state.load_image(path)
        except (FileNotFoundError, ValueError, TypeError) as error:
            QMessageBox.warning(
                self,
                "Unable to Open Image",
                str(error),
            )
            return

        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()

        self.statusBar().showMessage(
            f"Loaded: {path}",
            5000,
        )

    def undo(self):
        if not self.image_state.undo():
            return

        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()

    def redo(self):
        if not self.image_state.redo():
            return

        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()

    def reset(self):
        if not self.image_state.reset():
            return

        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()

    def _apply_preprocessing(
        self,
        operation: str,
        parameters: dict[str, object],
    ) -> None:
        """Process the full-resolution current image as one undoable action."""

        if not self.image_state.has_image:
            return

        operations = {
            "histogram": ("Histogram Equalization", equalize_histogram),
            "clahe": ("CLAHE", apply_clahe),
            "white_balance": ("Auto White Balance", auto_white_balance),
            "normalize": ("Normalize", normalize_image),
            "resize": ("Resize", resize_image),
        }

        try:
            label, processor = operations[operation]
            result = processor(self.image_state.current_image, **parameters)
            self.image_state.apply_change(result)
        except KeyError:
            QMessageBox.warning(
                self,
                "Processing Failed",
                f"Unknown preprocessing operation: {operation}",
            )
            return
        except (TypeError, ValueError, cv2.error) as error:
            QMessageBox.warning(self, "Processing Failed", str(error))
            return

        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()
        self.statusBar().showMessage(f"Applied: {label}", 5000)

    def _refresh_analysis(self):
        if not self.image_state.has_image:
            self.image_profile = None
            self.analysis_panel.clear()
            return

        self.image_profile = analyze_image(
            self.image_state.current_image
        )
        self.analysis_panel.update_profile(
            self.image_profile
        )

    def _refresh_viewer(self):
        if self.image_state.has_image:
            self.image_viewer.set_image(
                self.image_state.current_image
            )
        else:
            self.image_viewer.clear()

    def _update_action_states(self):
        has_image = self.image_state.has_image

        self.toolbar.open_action.setEnabled(True)
        self.toolbar.save_action.setEnabled(False)
        self.toolbar.undo_action.setEnabled(
            self.image_state.can_undo
        )
        self.toolbar.redo_action.setEnabled(
            self.image_state.can_redo
        )
        self.toolbar.reset_action.setEnabled(has_image)
        if has_image:
            height, width = self.image_state.current_image.shape[:2]
            self.prepare_panel.set_image_dimensions((width, height))
        else:
            self.prepare_panel.set_image_dimensions(None)
