"""Controls for guided Photo Repair mode."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class RepairPanel(QWidget):
    """Expose mask editing and inpainting controls without image algorithms."""

    tool_changed = Signal(str)
    repair_mode_changed = Signal(bool)
    preview_changed = Signal(bool)
    undo_mask_requested = Signal()
    redo_mask_requested = Signal()
    clear_mask_requested = Signal()
    inpaint_requested = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.addWidget(QLabel("PHOTO REPAIR"))

        self.repair_mode = QCheckBox("Enable repair painting")
        layout.addWidget(self.repair_mode)

        self.brush_button = QPushButton("Brush")
        self.eraser_button = QPushButton("Eraser")
        self.brush_button.setCheckable(True)
        self.eraser_button.setCheckable(True)
        self.brush_button.setChecked(True)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_group.addButton(self.brush_button)
        self._tool_group.addButton(self.eraser_button)
        tools = QHBoxLayout()
        tools.addWidget(self.brush_button)
        tools.addWidget(self.eraser_button)
        layout.addLayout(tools)

        layout.addWidget(QLabel("Brush Size"))
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 200)
        self.brush_size.setValue(15)
        layout.addWidget(self.brush_size)

        self.preview_mask = QCheckBox("Show mask overlay")
        self.preview_mask.setChecked(True)
        layout.addWidget(self.preview_mask)

        history = QHBoxLayout()
        self.undo_mask_button = QPushButton("Undo Mask")
        self.redo_mask_button = QPushButton("Redo Mask")
        history.addWidget(self.undo_mask_button)
        history.addWidget(self.redo_mask_button)
        layout.addLayout(history)
        self.clear_mask_button = QPushButton("Clear Mask")
        layout.addWidget(self.clear_mask_button)

        layout.addWidget(QLabel("Inpainting Method"))
        self.method = QComboBox()
        self.method.addItem("Telea", "telea")
        self.method.addItem("Navier-Stokes", "navier_stokes")
        layout.addWidget(self.method)
        layout.addWidget(QLabel("Repair Radius"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 100.0)
        self.radius.setSingleStep(0.5)
        self.radius.setValue(3.0)
        layout.addWidget(self.radius)
        self.apply_button = QPushButton("Apply Repair")
        self.apply_button.setMinimumHeight(36)
        layout.addWidget(self.apply_button)
        layout.addStretch()

        self.brush_button.clicked.connect(lambda: self.tool_changed.emit("brush"))
        self.eraser_button.clicked.connect(lambda: self.tool_changed.emit("eraser"))
        self.repair_mode.toggled.connect(self.repair_mode_changed)
        self.preview_mask.toggled.connect(self.preview_changed)
        self.undo_mask_button.clicked.connect(self.undo_mask_requested)
        self.redo_mask_button.clicked.connect(self.redo_mask_requested)
        self.clear_mask_button.clicked.connect(self.clear_mask_requested)
        self.apply_button.clicked.connect(self._request_inpaint)
        self.set_repair_available(False, False, False, False)

    def set_repair_available(
        self,
        image_available: bool,
        has_damage: bool,
        can_undo: bool,
        can_redo: bool,
    ) -> None:
        """Synchronize repair controls with mask state."""

        for control in (
            self.repair_mode,
            self.brush_button,
            self.eraser_button,
            self.brush_size,
            self.preview_mask,
            self.method,
            self.radius,
        ):
            control.setEnabled(image_available)
        self.clear_mask_button.setEnabled(has_damage)
        self.apply_button.setEnabled(has_damage)
        self.undo_mask_button.setEnabled(can_undo)
        self.redo_mask_button.setEnabled(can_redo)

    def _request_inpaint(self) -> None:
        self.inpaint_requested.emit(self.method.currentData(), self.radius.value())
