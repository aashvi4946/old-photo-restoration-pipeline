"""Inline controls for Phase 5 denoising operations."""

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


class DenoisePanel(QWidget):
    """Expose manual denoising actions with compact inline parameters."""

    operation_requested = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(9, 9, 9, 9)

        layout.addWidget(QLabel("DENOISE"))

        self.gaussian_button = QPushButton("Gaussian…")
        self.median_button = QPushButton("Median…")
        self.bilateral_button = QPushButton("Bilateral…")
        self.non_local_means_button = QPushButton("Non-Local Means…")

        for button in (
            self.gaussian_button,
            self.median_button,
            self.bilateral_button,
            self.non_local_means_button,
        ):
            button.setMinimumHeight(36)

        layout.addWidget(self.gaussian_button)
        self.gaussian_controls = self._create_gaussian_controls()
        layout.addWidget(self.gaussian_controls)
        layout.addWidget(self.median_button)
        self.median_controls = self._create_median_controls()
        layout.addWidget(self.median_controls)
        layout.addWidget(self.bilateral_button)
        self.bilateral_controls = self._create_bilateral_controls()
        layout.addWidget(self.bilateral_controls)
        layout.addWidget(self.non_local_means_button)
        self.non_local_means_controls = self._create_non_local_means_controls()
        layout.addWidget(self.non_local_means_controls)
        layout.addStretch()

        self.gaussian_button.clicked.connect(
            lambda: self._toggle_controls(self.gaussian_controls)
        )
        self.median_button.clicked.connect(
            lambda: self._toggle_controls(self.median_controls)
        )
        self.bilateral_button.clicked.connect(
            lambda: self._toggle_controls(self.bilateral_controls)
        )
        self.non_local_means_button.clicked.connect(
            lambda: self._toggle_controls(self.non_local_means_controls)
        )

        self.set_image_available(False)

    def set_image_available(self, available: bool) -> None:
        """Enable denoising actions only when an image is loaded."""

        for button in (
            self.gaussian_button,
            self.median_button,
            self.bilateral_button,
            self.non_local_means_button,
        ):
            button.setEnabled(available)

        if not available:
            self._hide_parameter_controls()

    def _create_gaussian_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.gaussian_kernel = self._odd_kernel_choice("5")
        layout.addRow("Kernel", self.gaussian_kernel)
        self.gaussian_sigma = self._positive_float_choice(0.0, minimum=0.0)
        layout.addRow("Sigma", self.gaussian_sigma)
        self._add_actions(layout, self._apply_gaussian)
        controls.hide()
        return controls

    def _create_median_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.median_kernel = self._odd_kernel_choice("3")
        layout.addRow("Kernel", self.median_kernel)
        self._add_actions(layout, self._apply_median)
        controls.hide()
        return controls

    def _create_bilateral_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.bilateral_diameter = QSpinBox()
        self.bilateral_diameter.setRange(1, 99)
        self.bilateral_diameter.setValue(9)
        layout.addRow("Diameter", self.bilateral_diameter)
        self.bilateral_sigma_color = self._positive_float_choice(75.0)
        layout.addRow("Color Sigma", self.bilateral_sigma_color)
        self.bilateral_sigma_space = self._positive_float_choice(75.0)
        layout.addRow("Space Sigma", self.bilateral_sigma_space)
        self._add_actions(layout, self._apply_bilateral)
        controls.hide()
        return controls

    def _create_non_local_means_controls(self) -> QWidget:
        controls = QWidget()
        layout = QFormLayout(controls)
        self.nlm_h = self._positive_float_choice(10.0)
        layout.addRow("Strength", self.nlm_h)
        self.nlm_h_color = self._positive_float_choice(10.0)
        layout.addRow("Color Strength", self.nlm_h_color)
        self.nlm_template_window = self._odd_window_choice("7")
        layout.addRow("Template Window", self.nlm_template_window)
        self.nlm_search_window = self._odd_window_choice("21")
        layout.addRow("Search Window", self.nlm_search_window)
        self._add_actions(layout, self._apply_non_local_means)
        controls.hide()
        return controls

    @staticmethod
    def _odd_kernel_choice(default: str) -> QComboBox:
        choice = QComboBox()
        choice.addItems(["3", "5", "7", "9"])
        choice.setCurrentText(default)
        return choice

    @staticmethod
    def _odd_window_choice(default: str) -> QComboBox:
        choice = QComboBox()
        choice.addItems(["3", "5", "7", "9", "11", "15", "21", "31"])
        choice.setCurrentText(default)
        return choice

    @staticmethod
    def _positive_float_choice(
        default: float,
        minimum: float = 0.1,
    ) -> QDoubleSpinBox:
        choice = QDoubleSpinBox()
        choice.setRange(minimum, 1000.0)
        choice.setSingleStep(1.0)
        choice.setValue(default)
        return choice

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
            self.gaussian_controls,
            self.median_controls,
            self.bilateral_controls,
            self.non_local_means_controls,
        ):
            controls.hide()

    def _emit_operation(self, operation: str, parameters: dict[str, object]) -> None:
        self.operation_requested.emit(operation, parameters)
        self._hide_parameter_controls()

    def _apply_gaussian(self) -> None:
        self._emit_operation(
            "gaussian",
            {
                "kernel_size": int(self.gaussian_kernel.currentText()),
                "sigma": self.gaussian_sigma.value(),
            },
        )

    def _apply_median(self) -> None:
        self._emit_operation(
            "median",
            {"kernel_size": int(self.median_kernel.currentText())},
        )

    def _apply_bilateral(self) -> None:
        self._emit_operation(
            "bilateral",
            {
                "diameter": self.bilateral_diameter.value(),
                "sigma_color": self.bilateral_sigma_color.value(),
                "sigma_space": self.bilateral_sigma_space.value(),
            },
        )

    def _apply_non_local_means(self) -> None:
        self._emit_operation(
            "non_local_means",
            {
                "h": self.nlm_h.value(),
                "h_color": self.nlm_h_color.value(),
                "template_window_size": int(self.nlm_template_window.currentText()),
                "search_window_size": int(self.nlm_search_window.currentText()),
            },
        )
