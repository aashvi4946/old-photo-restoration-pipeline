"""Inline controls for Phase 6 deblurring operations."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class DeblurPanel(QWidget):
    """Expose manual deblurring operations with inline parameter controls."""

    operation_requested = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(9, 9, 9, 9)

        layout.addWidget(QLabel("DEBLUR"))

        self.unsharp_button = QPushButton("Unsharp Mask…")
        self.wiener_button = QPushButton("Wiener…")
        self.lucy_richardson_button = QPushButton("Lucy–Richardson…")

        for button in (
            self.unsharp_button,
            self.wiener_button,
            self.lucy_richardson_button,
        ):
            button.setMinimumHeight(36)

        layout.addWidget(self.unsharp_button)
        self.unsharp_controls = self._create_unsharp_controls()
        layout.addWidget(self.unsharp_controls)
        layout.addWidget(self.wiener_button)
        self.wiener_controls = self._create_wiener_controls()
        layout.addWidget(self.wiener_controls)
        layout.addWidget(self.lucy_richardson_button)
        self.lucy_richardson_controls = self._create_lucy_richardson_controls()
        layout.addWidget(self.lucy_richardson_controls)
        layout.addStretch()

        self.unsharp_button.clicked.connect(
            lambda: self._toggle_controls(self.unsharp_controls)
        )
        self.wiener_button.clicked.connect(
            lambda: self._toggle_controls(self.wiener_controls)
        )
        self.lucy_richardson_button.clicked.connect(
            lambda: self._toggle_controls(self.lucy_richardson_controls)
        )

        self.set_image_available(False)

    def set_image_available(self, available: bool) -> None:
        """Enable deblurring actions only while an image is loaded."""

        for button in (
            self.unsharp_button,
            self.wiener_button,
            self.lucy_richardson_button,
        ):
            button.setEnabled(available)

        if not available:
            self._hide_parameter_controls()

    def _create_unsharp_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.unsharp_radius = self._float_input(1.0, 0.1, 100.0)
        layout.addRow("Radius", self.unsharp_radius)
        self.unsharp_amount = self._float_input(1.0, 0.1, 10.0)
        layout.addRow("Amount", self.unsharp_amount)
        self.unsharp_threshold = QSpinBox()
        self.unsharp_threshold.setRange(0, 255)
        layout.addRow("Threshold", self.unsharp_threshold)
        self._add_actions(layout, self._apply_unsharp)
        controls.hide()
        return controls

    def _create_wiener_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.wiener_psf_length = self._psf_length_choice("9")
        layout.addRow("PSF Length", self.wiener_psf_length)
        self.wiener_psf_angle = self._angle_input()
        layout.addRow("PSF Angle", self.wiener_psf_angle)
        self.wiener_balance = self._float_input(0.01, 0.001, 1.0, 0.001)
        layout.addRow("Balance", self.wiener_balance)
        self._add_actions(layout, self._apply_wiener)
        controls.hide()
        return controls

    def _create_lucy_richardson_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.lucy_psf_length = self._psf_length_choice("9")
        layout.addRow("PSF Length", self.lucy_psf_length)
        self.lucy_psf_angle = self._angle_input()
        layout.addRow("PSF Angle", self.lucy_psf_angle)
        self.lucy_iterations = QSpinBox()
        self.lucy_iterations.setRange(1, 100)
        self.lucy_iterations.setValue(10)
        layout.addRow("Iterations", self.lucy_iterations)
        self._add_actions(layout, self._apply_lucy_richardson)
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

    @staticmethod
    def _psf_length_choice(default: str) -> QComboBox:
        choice = QComboBox()
        choice.addItems(["3", "5", "7", "9", "15", "21"])
        choice.setCurrentText(default)
        return choice

    @staticmethod
    def _angle_input() -> QDoubleSpinBox:
        input_field = QDoubleSpinBox()
        input_field.setRange(-360.0, 360.0)
        input_field.setSingleStep(1.0)
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
            self.unsharp_controls,
            self.wiener_controls,
            self.lucy_richardson_controls,
        ):
            controls.hide()

    def _emit_operation(self, operation: str, parameters: dict[str, object]) -> None:
        self.operation_requested.emit(operation, parameters)
        self._hide_parameter_controls()

    def _apply_unsharp(self) -> None:
        self._emit_operation(
            "unsharp",
            {
                "radius": self.unsharp_radius.value(),
                "amount": self.unsharp_amount.value(),
                "threshold": self.unsharp_threshold.value(),
            },
        )

    def _apply_wiener(self) -> None:
        self._emit_operation(
            "wiener",
            {
                "psf_length": int(self.wiener_psf_length.currentText()),
                "psf_angle": self.wiener_psf_angle.value(),
                "balance": self.wiener_balance.value(),
            },
        )

    def _apply_lucy_richardson(self) -> None:
        self._emit_operation(
            "lucy_richardson",
            {
                "psf_length": int(self.lucy_psf_length.currentText()),
                "psf_angle": self.lucy_psf_angle.value(),
                "iterations": self.lucy_iterations.value(),
            },
        )
