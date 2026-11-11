"""Manual pipeline controls for Phase 10 Advanced mode."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from core.pipeline import OPERATION_DEFINITIONS, ParameterSpec, ProcessingPipeline


class AdvancedPanel(QWidget):
    """Build an ordered manual processing pipeline without embedding algorithms."""

    pipeline_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = ProcessingPipeline()
        self._parameter_widgets: dict[str, QWidget] = {}
        self._image_available = False
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.addWidget(QLabel("ADVANCED"))
        layout.addWidget(QLabel("Operation Browser"))
        self.operation_browser = QComboBox()
        for definition in OPERATION_DEFINITIONS:
            self.operation_browser.addItem(f"{definition.category} — {definition.label}", definition.identifier)
        layout.addWidget(self.operation_browser)
        self.add_button = QPushButton("Add to Pipeline")
        layout.addWidget(self.add_button)
        layout.addWidget(QLabel("Pipeline"))
        self.pipeline_list = QListWidget()
        self.pipeline_list.setMinimumHeight(120)
        layout.addWidget(self.pipeline_list)
        movement = QHBoxLayout()
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        self.remove_button = QPushButton("Remove")
        for button in (self.move_up_button, self.move_down_button, self.remove_button):
            movement.addWidget(button)
        layout.addLayout(movement)
        self.clear_button = QPushButton("Clear Pipeline")
        layout.addWidget(self.clear_button)
        layout.addWidget(QLabel("Parameters"))
        self.parameters_widget = QWidget()
        self.parameters_layout = QFormLayout(self.parameters_widget)
        layout.addWidget(self.parameters_widget)
        self.update_parameters_button = QPushButton("Update Parameters")
        self.apply_button = QPushButton("Apply Pipeline")
        self.apply_button.setMinimumHeight(36)
        layout.addWidget(self.update_parameters_button)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        self.add_button.clicked.connect(self._add_operation)
        self.pipeline_list.currentRowChanged.connect(self._select_step)
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button.clicked.connect(self._clear_pipeline)
        self.update_parameters_button.clicked.connect(self._update_parameters)
        self.apply_button.clicked.connect(lambda: self.pipeline_requested.emit(self.pipeline))
        self.set_image_available(False)

    def set_image_available(self, available: bool) -> None:
        self.operation_browser.setEnabled(available)
        self.add_button.setEnabled(available)
        self.pipeline_list.setEnabled(available)
        self._image_available = available
        self._refresh_actions()

    def _add_operation(self) -> None:
        self.pipeline.add(self.operation_browser.currentData())
        self._refresh_pipeline(len(self.pipeline.steps) - 1)

    def _select_step(self, index: int) -> None:
        self._rebuild_parameter_controls(index)
        self._refresh_actions()

    def _move_selected(self, offset: int) -> None:
        try:
            target = self.pipeline.move(self.pipeline_list.currentRow(), offset)
        except ValueError:
            return
        self._refresh_pipeline(target)

    def _remove_selected(self) -> None:
        index = self.pipeline_list.currentRow()
        try:
            self.pipeline.remove(index)
        except ValueError:
            return
        self._refresh_pipeline(min(index, len(self.pipeline.steps) - 1))

    def _clear_pipeline(self) -> None:
        self.pipeline.clear()
        self._refresh_pipeline()

    def _update_parameters(self) -> None:
        index = self.pipeline_list.currentRow()
        if index < 0:
            return
        parameters = {name: self._parameter_value(widget) for name, widget in self._parameter_widgets.items()}
        try:
            self.pipeline.update_parameters(index, parameters)
        except (TypeError, ValueError):
            return
        self._refresh_pipeline(index)

    def _refresh_pipeline(self, select_index: int = -1) -> None:
        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()
        for number, step in enumerate(self.pipeline.steps, start=1):
            definition = self.pipeline.definition_for(step.operation)
            self.pipeline_list.addItem(f"{number}. {definition.label}")
        self.pipeline_list.blockSignals(False)
        if select_index >= 0 and self.pipeline.steps:
            self.pipeline_list.setCurrentRow(select_index)
        else:
            self._rebuild_parameter_controls(-1)
        self._refresh_actions()

    def _rebuild_parameter_controls(self, index: int) -> None:
        while self.parameters_layout.count():
            item = self.parameters_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._parameter_widgets = {}
        if not 0 <= index < len(self.pipeline.steps):
            return
        step = self.pipeline.steps[index]
        definition = self.pipeline.definition_for(step.operation)
        if not definition.parameters:
            self.parameters_layout.addRow(QLabel("This operation has no parameters."))
            return
        for spec in definition.parameters:
            widget = self._create_parameter_widget(spec, step.parameters[spec.name])
            self._parameter_widgets[spec.name] = widget
            self.parameters_layout.addRow(spec.label, widget)

    @staticmethod
    def _create_parameter_widget(spec: ParameterSpec, value: object) -> QWidget:
        if spec.kind == "int":
            widget = QSpinBox()
            widget.setRange(int(spec.minimum), int(spec.maximum))
            widget.setSingleStep(int(spec.step))
            widget.setValue(int(value))
            return widget
        if spec.kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(spec.minimum), float(spec.maximum))
            widget.setDecimals(3)
            widget.setSingleStep(float(spec.step))
            widget.setValue(float(value))
            return widget
        if spec.kind == "choice":
            widget = QComboBox()
            for label, data in spec.choices:
                widget.addItem(label, data)
            widget.setCurrentIndex(widget.findData(value))
            return widget
        raise ValueError(f"Unsupported parameter kind: {spec.kind}")

    @staticmethod
    def _parameter_value(widget: QWidget) -> object:
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        raise TypeError("Unsupported parameter widget.")

    def _refresh_actions(self) -> None:
        index = self.pipeline_list.currentRow()
        selected = self._image_available and 0 <= index < len(self.pipeline.steps)
        self.move_up_button.setEnabled(selected and index > 0)
        self.move_down_button.setEnabled(selected and index < len(self.pipeline.steps) - 1)
        self.remove_button.setEnabled(selected)
        self.clear_button.setEnabled(self._image_available and not self.pipeline.is_empty)
        self.update_parameters_button.setEnabled(selected)
        self.apply_button.setEnabled(self._image_available and not self.pipeline.is_empty)
