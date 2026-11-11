"""Inline controls for Phase 8 final image refinement."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FinishPanel(QWidget):
    """Expose manual finishing actions with compact inline parameters."""

    operation_requested = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.addWidget(QLabel("FINISH"))

        self.sharpen_button = QPushButton("Sharpen…")
        self.edge_button = QPushButton("Edge Enhancement…")
        self.brightness_button = QPushButton("Brightness…")
        self.contrast_button = QPushButton("Contrast…")
        self.gamma_button = QPushButton("Gamma…")
        self.color_correction_button = QPushButton("Color Correction…")
        self.black_white_button = QPushButton("B&W")
        self.vintage_button = QPushButton("Vintage…")

        self._buttons = (
            self.sharpen_button,
            self.edge_button,
            self.brightness_button,
            self.contrast_button,
            self.gamma_button,
            self.color_correction_button,
            self.black_white_button,
            self.vintage_button,
        )
        for button in self._buttons:
            button.setMinimumHeight(36)

        layout.addWidget(self.sharpen_button)
        self.sharpen_controls = self._create_sharpen_controls()
        layout.addWidget(self.sharpen_controls)
        layout.addWidget(self.edge_button)
        self.edge_controls = self._create_edge_controls()
        layout.addWidget(self.edge_controls)
        layout.addWidget(self.brightness_button)
        self.brightness_controls = self._create_brightness_controls()
        layout.addWidget(self.brightness_controls)
        layout.addWidget(self.contrast_button)
        self.contrast_controls = self._create_contrast_controls()
        layout.addWidget(self.contrast_controls)
        layout.addWidget(self.gamma_button)
        self.gamma_controls = self._create_gamma_controls()
        layout.addWidget(self.gamma_controls)
        layout.addWidget(self.color_correction_button)
        self.color_correction_controls = self._create_color_correction_controls()
        layout.addWidget(self.color_correction_controls)
        layout.addWidget(self.black_white_button)
        layout.addWidget(self.vintage_button)
        self.vintage_controls = self._create_vintage_controls()
        layout.addWidget(self.vintage_controls)
        layout.addStretch()

        self.sharpen_button.clicked.connect(
            lambda: self._toggle_controls(self.sharpen_controls)
        )
        self.edge_button.clicked.connect(lambda: self._toggle_controls(self.edge_controls))
        self.brightness_button.clicked.connect(
            lambda: self._toggle_controls(self.brightness_controls)
        )
        self.contrast_button.clicked.connect(
            lambda: self._toggle_controls(self.contrast_controls)
        )
        self.gamma_button.clicked.connect(lambda: self._toggle_controls(self.gamma_controls))
        self.color_correction_button.clicked.connect(
            lambda: self._toggle_controls(self.color_correction_controls)
        )
        self.black_white_button.clicked.connect(
            lambda: self.operation_requested.emit("black_white", {})
        )
        self.vintage_button.clicked.connect(
            lambda: self._toggle_controls(self.vintage_controls)
        )
        self.set_image_available(False)

    def set_image_available(self, available: bool) -> None:
        """Enable finishing actions only when an image is loaded."""

        for button in self._buttons:
            button.setEnabled(available)
        if not available:
            self._hide_parameter_controls()

    def _create_sharpen_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.sharpen_amount = self._float_input(1.0, 0.0, 3.0)
        layout.addRow("Amount", self.sharpen_amount)
        self._add_actions(layout, self._apply_sharpen)
        controls.hide()
        return controls

    def _create_edge_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.edge_strength = self._float_input(0.5, 0.0, 3.0)
        layout.addRow("Strength", self.edge_strength)
        self._add_actions(layout, self._apply_edge_enhancement)
        controls.hide()
        return controls

    def _create_brightness_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.brightness_value = QSpinBox()
        self.brightness_value.setRange(-255, 255)
        layout.addRow("Value", self.brightness_value)
        self._add_actions(layout, self._apply_brightness)
        controls.hide()
        return controls

    def _create_contrast_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.contrast_factor = self._float_input(1.0, 0.1, 3.0)
        layout.addRow("Factor", self.contrast_factor)
        self._add_actions(layout, self._apply_contrast)
        controls.hide()
        return controls

    def _create_gamma_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.gamma_value = self._float_input(1.0, 0.1, 5.0)
        layout.addRow("Gamma", self.gamma_value)
        self._add_actions(layout, self._apply_gamma)
        controls.hide()
        return controls

    def _create_color_correction_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.blue_gain = self._float_input(1.0, 0.1, 3.0)
        self.green_gain = self._float_input(1.0, 0.1, 3.0)
        self.red_gain = self._float_input(1.0, 0.1, 3.0)
        layout.addRow("Blue Gain", self.blue_gain)
        layout.addRow("Green Gain", self.green_gain)
        layout.addRow("Red Gain", self.red_gain)
        self._add_actions(layout, self._apply_color_correction)
        controls.hide()
        return controls

    def _create_vintage_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.vintage_intensity = self._float_input(0.35, 0.0, 1.0, 0.05)
        layout.addRow("Intensity", self.vintage_intensity)
        self._add_actions(layout, self._apply_vintage)
        controls.hide()
        return controls

    @staticmethod
    def _float_input(
        default: float,
        minimum: float,
        maximum: float,
        step: float = 0.1,
    ) -> QDoubleSpinBox:
        input_field = QDoubleSpinBox()
        input_field.setRange(minimum, maximum)
        input_field.setSingleStep(step)
        input_field.setValue(default)
        return input_field

    def _add_actions(self, layout: QFormLayout, apply) -> None:
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        apply_button.clicked.connect(apply)
        cancel_button.clicked.connect(self._hide_parameter_controls)
        layout.addRow(apply_button, cancel_button)

    def _toggle_controls(self, controls: QWidget) -> None:
        should_show = controls.isHidden()
        self._hide_parameter_controls()
        controls.setVisible(should_show)

    def _hide_parameter_controls(self) -> None:
        for controls in (
            self.sharpen_controls,
            self.edge_controls,
            self.brightness_controls,
            self.contrast_controls,
            self.gamma_controls,
            self.color_correction_controls,
            self.vintage_controls,
        ):
            controls.hide()

    def _emit_operation(self, operation: str, parameters: dict[str, object]) -> None:
        self.operation_requested.emit(operation, parameters)
        self._hide_parameter_controls()

    def _apply_sharpen(self) -> None:
        self._emit_operation("sharpen", {"amount": self.sharpen_amount.value()})

    def _apply_edge_enhancement(self) -> None:
        self._emit_operation("edge_enhancement", {"strength": self.edge_strength.value()})

    def _apply_brightness(self) -> None:
        self._emit_operation("brightness", {"value": self.brightness_value.value()})

    def _apply_contrast(self) -> None:
        self._emit_operation("contrast", {"factor": self.contrast_factor.value()})

    def _apply_gamma(self) -> None:
        self._emit_operation("gamma", {"gamma": self.gamma_value.value()})

    def _apply_color_correction(self) -> None:
        self._emit_operation(
            "color_correction",
            {
                "blue_gain": self.blue_gain.value(),
                "green_gain": self.green_gain.value(),
                "red_gain": self.red_gain.value(),
            },
        )

    def _apply_vintage(self) -> None:
        self._emit_operation("vintage", {"intensity": self.vintage_intensity.value()})
