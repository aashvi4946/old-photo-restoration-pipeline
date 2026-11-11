import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QScrollArea,
)
from core.image_state import ImageState
from core.repair_mask_state import RepairMaskState
from processing.analysis.analyze import analyze_image
from processing.preprocessing.clahe import apply_clahe
from processing.preprocessing.histogram import equalize_histogram
from processing.preprocessing.normalization import normalize_image
from processing.preprocessing.resize import resize_image
from processing.preprocessing.white_balance import auto_white_balance
from processing.restoration.denoise import (
    apply_bilateral_denoise,
    apply_gaussian_denoise,
    apply_median_denoise,
    apply_non_local_means_denoise,
)
from processing.restoration.deblur import (
    apply_lucy_richardson_deconvolution,
    apply_unsharp_mask,
    apply_wiener_deconvolution,
    create_motion_psf,
)
from processing.restoration.inpaint import apply_inpainting
from models.image_profile import ImageProfile
from ui.toolbar import AppToolbar
from ui.image_viewer import ImageViewer
from ui.analysis_panel import AnalysisPanel
from ui.prepare_panel import PreparePanel
from ui.denoise_panel import DenoisePanel
from ui.deblur_panel import DeblurPanel
from ui.repair_panel import RepairPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Old Photo Restorer")
        self.resize(1200, 800)

        self.image_state = ImageState()
        self.repair_mask_state = RepairMaskState()
        self.image_profile: ImageProfile | None = None
        self._repair_tool = "brush"
        self._repair_mode_enabled = False

        self._setup_ui()
        self._connect_actions()
        self._update_action_states()

    def _setup_ui(self):
        self.toolbar = AppToolbar(self)
        self.addToolBar(self.toolbar)

        self.image_viewer = ImageViewer()
        self.analysis_panel = AnalysisPanel()
        self.prepare_panel = PreparePanel()
        self.denoise_panel = DenoisePanel()
        self.deblur_panel = DeblurPanel()
        self.repair_panel = RepairPanel()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.prepare_panel)
        left_layout.addWidget(self.denoise_panel)
        left_layout.addWidget(self.deblur_panel)
        left_layout.addWidget(self.repair_panel)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)

        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(left_scroll, 0)
        layout.addWidget(self.image_viewer, 1)
        layout.addWidget(self.analysis_panel, 0)

        self.setCentralWidget(central_widget)
        
    def _connect_actions(self):
        self.toolbar.open_action.triggered.connect(self.open_image)
        self.toolbar.undo_action.triggered.connect(self.undo)
        self.toolbar.redo_action.triggered.connect(self.redo)
        self.toolbar.reset_action.triggered.connect(self.reset)
        self.prepare_panel.operation_requested.connect(self._apply_preprocessing)
        self.denoise_panel.operation_requested.connect(self._apply_denoising)
        self.deblur_panel.operation_requested.connect(self._apply_deblurring)
        self.repair_panel.tool_changed.connect(self._set_repair_tool)
        self.repair_panel.repair_mode_changed.connect(self._set_repair_mode)
        self.repair_panel.preview_changed.connect(self._set_repair_preview)
        self.repair_panel.undo_mask_requested.connect(self._undo_repair_mask)
        self.repair_panel.redo_mask_requested.connect(self._redo_repair_mask)
        self.repair_panel.clear_mask_requested.connect(self._clear_repair_mask)
        self.repair_panel.inpaint_requested.connect(self._apply_repair)
        self.image_viewer.repair_stroke_started.connect(self._begin_repair_stroke)
        self.image_viewer.repair_stroke_extended.connect(self._extend_repair_stroke)
        self.image_viewer.repair_stroke_finished.connect(self._end_repair_stroke)

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

        self.repair_mask_state.reset()

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
        self.repair_mask_state.reset()
        self._update_action_states()

    def _apply_preprocessing(
        self,
        operation: str,
        parameters: dict[str, object],
    ) -> None:
        """Process the full-resolution current image as one undoable action."""

        operations = {
            "histogram": ("Histogram Equalization", equalize_histogram),
            "clahe": ("CLAHE", apply_clahe),
            "white_balance": ("Auto White Balance", auto_white_balance),
            "normalize": ("Normalize", normalize_image),
            "resize": ("Resize", resize_image),
        }

        self._apply_operation(operation, parameters, operations)

    def _apply_denoising(
        self,
        operation: str,
        parameters: dict[str, object],
    ) -> None:
        """Apply an explicit denoising operation as one undoable action."""

        operations = {
            "gaussian": ("Gaussian Denoise", apply_gaussian_denoise),
            "median": ("Median Denoise", apply_median_denoise),
            "bilateral": ("Bilateral Denoise", apply_bilateral_denoise),
            "non_local_means": (
                "Non-Local Means Denoise",
                apply_non_local_means_denoise,
            ),
        }
        self._apply_operation(operation, parameters, operations)

    def _apply_deblurring(
        self,
        operation: str,
        parameters: dict[str, object],
    ) -> None:
        """Apply an explicit deblurring operation as one undoable action."""

        operations = {
            "unsharp": ("Unsharp Mask", apply_unsharp_mask),
            "wiener": ("Wiener Deconvolution", apply_wiener_deconvolution),
            "lucy_richardson": (
                "Lucy–Richardson Deconvolution",
                apply_lucy_richardson_deconvolution,
            ),
        }
        operation_parameters = dict(parameters)

        if operation in {"wiener", "lucy_richardson"}:
            try:
                psf_length = operation_parameters.pop("psf_length")
                psf_angle = operation_parameters.pop("psf_angle")
                operation_parameters["psf"] = create_motion_psf(
                    length=psf_length,
                    angle=psf_angle,
                )
            except (KeyError, TypeError, ValueError) as error:
                QMessageBox.warning(self, "Processing Failed", str(error))
                return

        self._apply_operation(operation, operation_parameters, operations)

    def _apply_operation(
        self,
        operation: str,
        parameters: dict[str, object],
        operations: dict[str, tuple[str, object]],
    ) -> None:
        """Apply a processing function and refresh UI only after success."""

        if not self.image_state.has_image:
            return

        try:
            label, processor = operations[operation]
            result = processor(self.image_state.current_image, **parameters)
            self.image_state.apply_change(result)
        except KeyError:
            QMessageBox.warning(
                self,
                "Processing Failed",
                f"Unknown processing operation: {operation}",
            )
            return
        except (TypeError, ValueError, cv2.error) as error:
            QMessageBox.warning(self, "Processing Failed", str(error))
            return

        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()
        self.statusBar().showMessage(f"Applied: {label}", 5000)

    def _set_repair_tool(self, tool: str) -> None:
        """Select mask painting or erasing for the repair canvas."""

        self._repair_tool = tool

    def _set_repair_mode(self, enabled: bool) -> None:
        self._repair_mode_enabled = enabled
        self._refresh_repair_canvas()

    def _set_repair_preview(self, enabled: bool) -> None:
        self._refresh_repair_canvas()

    def _begin_repair_stroke(self, _point: tuple[int, int]) -> None:
        if self.image_state.has_image:
            self.repair_mask_state.begin_stroke()

    def _extend_repair_stroke(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> None:
        if not self.image_state.has_image:
            return
        self.repair_mask_state.paint_line(
            start,
            end,
            self.repair_panel.brush_size.value(),
            erase=self._repair_tool == "eraser",
        )
        self._refresh_repair_canvas()
        self._update_repair_actions()

    def _end_repair_stroke(self) -> None:
        self.repair_mask_state.end_stroke()
        self._refresh_repair_canvas()
        self._update_repair_actions()

    def _undo_repair_mask(self) -> None:
        if self.repair_mask_state.undo():
            self._refresh_repair_canvas()
            self._update_repair_actions()

    def _redo_repair_mask(self) -> None:
        if self.repair_mask_state.redo():
            self._refresh_repair_canvas()
            self._update_repair_actions()

    def _clear_repair_mask(self) -> None:
        if self.repair_mask_state.clear_all():
            self._refresh_repair_canvas()
            self._update_repair_actions()

    def _apply_repair(self, method: str, radius: float) -> None:
        """Commit a successful inpainting result and then clear its repair mask."""

        if not self.image_state.has_image:
            return
        try:
            repaired = apply_inpainting(
                self.image_state.current_image,
                self.repair_mask_state.mask,
                method=method,
                radius=radius,
            )
            self.image_state.apply_change(repaired)
        except (TypeError, ValueError, cv2.error) as error:
            QMessageBox.warning(self, "Repair Failed", str(error))
            return

        self.repair_mask_state.reset()
        self._refresh_analysis()
        self._refresh_viewer()
        self._update_action_states()
        self.statusBar().showMessage("Applied: Photo Repair", 5000)

    def _refresh_repair_canvas(self) -> None:
        self.image_viewer.set_repair_mask(self.repair_mask_state.mask)
        self.image_viewer.set_repair_mode(
            self._repair_mode_enabled and self.image_state.has_image,
            self.repair_panel.preview_mask.isChecked(),
        )

    def _update_repair_actions(self) -> None:
        self.repair_panel.set_repair_available(
            self.image_state.has_image,
            self.repair_mask_state.has_damage,
            self.repair_mask_state.can_undo,
            self.repair_mask_state.can_redo,
        )

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
        self.denoise_panel.set_image_available(has_image)
        self.deblur_panel.set_image_available(has_image)
        if has_image:
            height, width = self.image_state.current_image.shape[:2]
            self.repair_mask_state.set_image_shape(height, width)
        self._update_repair_actions()
        self._refresh_repair_canvas()
        if has_image:
            height, width = self.image_state.current_image.shape[:2]
            self.prepare_panel.set_image_dimensions((width, height))
        else:
            self.prepare_panel.set_image_dimensions(None)
