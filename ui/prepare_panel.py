"""Inline controls for Phase 4 preprocessing."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PreparePanel(QWidget):
    """Expose explicit preprocessing actions and inline parameter controls."""

    operation_requested = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_dimensions: tuple[int, int] | None = None
        self._resize_aspect_ratio = 1.0
        self._updating_resize_dimensions = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(9, 9, 9, 9)

        title = QLabel("PREPARE")
        title.setObjectName("prepareTitle")
        layout.addWidget(title)

        self.histogram_button = QPushButton("Histogram Equalization")
        self.clahe_button = QPushButton("CLAHE…")
        self.white_balance_button = QPushButton("Auto White Balance")
        self.normalize_button = QPushButton("Normalize")
        self.resize_button = QPushButton("Resize…")

        for button in (
            self.histogram_button,
            self.clahe_button,
            self.white_balance_button,
            self.normalize_button,
            self.resize_button,
        ):
            button.setMinimumHeight(36)

        layout.addWidget(self.histogram_button)
        layout.addWidget(self.clahe_button)
        self.clahe_controls = self._create_clahe_controls()
        layout.addWidget(self.clahe_controls)
        layout.addWidget(self.white_balance_button)
        layout.addWidget(self.normalize_button)
        layout.addWidget(self.resize_button)
        self.resize_controls = self._create_resize_controls()
        layout.addWidget(self.resize_controls)
        layout.addStretch()

        self.histogram_button.clicked.connect(
            lambda: self.operation_requested.emit("histogram", {})
        )
        self.white_balance_button.clicked.connect(
            lambda: self.operation_requested.emit("white_balance", {})
        )
        self.normalize_button.clicked.connect(
            lambda: self.operation_requested.emit("normalize", {})
        )
        self.clahe_button.clicked.connect(
            lambda: self._toggle_controls(self.clahe_controls)
        )
        self.resize_button.clicked.connect(
            lambda: self._toggle_controls(self.resize_controls)
        )

        self.set_image_dimensions(None)

    def set_image_dimensions(self, dimensions: tuple[int, int] | None) -> None:
        """Enable controls only when a full-resolution image is available."""

        self._image_dimensions = dimensions
        enabled = dimensions is not None
        for button in (
            self.histogram_button,
            self.clahe_button,
            self.white_balance_button,
            self.normalize_button,
            self.resize_button,
        ):
            button.setEnabled(enabled)

        if dimensions is None:
            self._hide_parameter_controls()
            return

        width, height = dimensions
        self._resize_aspect_ratio = width / height
        self._set_resize_dimensions(width, height)

    def _create_clahe_controls(self) -> QWidget:
        """Create the inline CLAHE parameter section."""

        controls = QWidget()
        layout = QFormLayout(controls)

        self.clahe_clip_limit = QDoubleSpinBox()
        self.clahe_clip_limit.setRange(0.1, 100.0)
        self.clahe_clip_limit.setSingleStep(0.1)
        self.clahe_clip_limit.setValue(2.0)
        layout.addRow("Clip Limit", self.clahe_clip_limit)

        self.clahe_tile_width = QSpinBox()
        self.clahe_tile_width.setRange(1, 256)
        self.clahe_tile_width.setValue(8)
        layout.addRow("Tile Width", self.clahe_tile_width)

        self.clahe_tile_height = QSpinBox()
        self.clahe_tile_height.setRange(1, 256)
        self.clahe_tile_height.setValue(8)
        layout.addRow("Tile Height", self.clahe_tile_height)

        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(self._apply_clahe)
        cancel_button.clicked.connect(self._hide_parameter_controls)
        layout.addRow(apply_button, cancel_button)
        controls.hide()

        return controls

    def _create_resize_controls(self) -> QWidget:
        """Create the inline resize parameter section."""

        controls = QWidget()
        layout = QFormLayout(controls)

        self.resize_width_input = QSpinBox()
        self.resize_width_input.setRange(1, 100_000)
        layout.addRow("Width", self.resize_width_input)

        self.resize_height_input = QSpinBox()
        self.resize_height_input.setRange(1, 100_000)
        layout.addRow("Height", self.resize_height_input)

        self.maintain_aspect_ratio = QCheckBox("Maintain aspect ratio")
        self.maintain_aspect_ratio.setChecked(True)
        layout.addRow(self.maintain_aspect_ratio)

        self.resize_width_input.valueChanged.connect(self._update_resize_height)
        self.resize_height_input.valueChanged.connect(self._update_resize_width)

        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(self._apply_resize)
        cancel_button.clicked.connect(self._hide_parameter_controls)
        layout.addRow(apply_button, cancel_button)
        controls.hide()

        return controls

    def _toggle_controls(self, controls: QWidget) -> None:
        """Show one parameter section directly below its corresponding button."""

        should_show = controls.isHidden()
        self._hide_parameter_controls()
        controls.setVisible(should_show)

    def _hide_parameter_controls(self) -> None:
        """Collapse all inline parameter sections."""

        self.clahe_controls.hide()
        self.resize_controls.hide()

    def _apply_clahe(self) -> None:
        self.operation_requested.emit(
            "clahe",
            {
                "clip_limit": self.clahe_clip_limit.value(),
                "tile_grid_size": (
                    self.clahe_tile_width.value(),
                    self.clahe_tile_height.value(),
                ),
            },
        )
        self._hide_parameter_controls()

    def _apply_resize(self) -> None:
        self.operation_requested.emit(
            "resize",
            {
                "width": self.resize_width_input.value(),
                "height": self.resize_height_input.value(),
            },
        )
        self._hide_parameter_controls()

    def _set_resize_dimensions(self, width: int, height: int) -> None:
        """Set inline resize values without triggering aspect-ratio updates."""

        with QSignalBlocker(self.resize_width_input):
            self.resize_width_input.setValue(width)
        with QSignalBlocker(self.resize_height_input):
            self.resize_height_input.setValue(height)

    def _update_resize_height(self, width: int) -> None:
        if self._updating_resize_dimensions or not self.maintain_aspect_ratio.isChecked():
            return

        self._updating_resize_dimensions = True
        with QSignalBlocker(self.resize_height_input):
            self.resize_height_input.setValue(
                max(1, round(width / self._resize_aspect_ratio))
            )
        self._updating_resize_dimensions = False

    def _update_resize_width(self, height: int) -> None:
        if self._updating_resize_dimensions or not self.maintain_aspect_ratio.isChecked():
            return

        self._updating_resize_dimensions = True
        with QSignalBlocker(self.resize_width_input):
            self.resize_width_input.setValue(
                max(1, round(height * self._resize_aspect_ratio))
            )
        self._updating_resize_dimensions = False
