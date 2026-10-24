"""参数面板：由 ParamSpec schema 自动生成控件。"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bac_count.core.models import ParamSpec


class _SliderRow(QWidget):
    """滑杆 + 数字框联动。"""

    valueChanged = Signal()

    def __init__(self, spec: ParamSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.is_float = spec.kind == "float"
        self.factor = 100 if self.is_float else 1
        lo = int((spec.min if spec.min is not None else 0) * self.factor)
        hi = int((spec.max if spec.max is not None else 100) * self.factor)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        if spec.step:
            self.slider.setSingleStep(max(1, int(spec.step * self.factor)))
        if self.is_float:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(2)
            self.spin.setRange(spec.min or 0.0, spec.max or 1e6)
            self.spin.setSingleStep(spec.step or 0.1)
        else:
            self.spin = QSpinBox()
            self.spin.setRange(int(spec.min or 0), int(spec.max or 1e6))
            self.spin.setSingleStep(int(spec.step or 1))
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 2)
        row.addWidget(self.spin, 1)
        self.set_value(spec.default)

    def _from_slider(self, v: int) -> None:
        val = v / self.factor if self.is_float else v
        if val != self.spin.value():
            self.spin.setValue(val)

    def _from_spin(self, val: float) -> None:
        v = int(round(val * self.factor))
        if v != self.slider.value():
            self.slider.setValue(v)
        self.valueChanged.emit()

    def value(self) -> Any:
        return float(self.spin.value()) if self.is_float else int(self.spin.value())

    def set_value(self, val: Any) -> None:
        self.blockSignals(True)
        self.spin.setValue(val)
        self.slider.setValue(int(round(float(val) * self.factor)))
        self.blockSignals(False)


class ParamPanel(QScrollArea):
    valueChanged = Signal(str, object)  # key, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self.setWidget(self._container)
        self._widgets: dict[str, tuple[ParamSpec, QWidget]] = {}
        self._loading = False

    def set_schema(self, specs: list[ParamSpec], values: dict[str, Any]) -> None:
        """重建控件；values 提供当前值（缺省用 spec.default）。"""
        self._loading = True
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()
        groups: dict[str, QVBoxLayout] = {}
        for spec in specs:
            if spec.group not in groups:
                box = QGroupBox(spec.group)
                gl = QFormLayout(box)
                gl.setLabelAlignment(Qt.AlignLeft)
                groups[spec.group] = gl
                self._layout.addWidget(box)
            val = values.get(spec.key, spec.default)
            w = self._make_widget(spec, val)
            groups[spec.group].addRow(QLabel(spec.label), w)
            self._widgets[spec.key] = (spec, w)
        self._layout.addStretch(1)
        self._loading = False

    def _make_widget(self, spec: ParamSpec, val: Any) -> QWidget:
        if spec.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(val))
            w.toggled.connect(lambda checked, k=spec.key: self._emit(k, checked))
        elif spec.kind == "combo":
            w = QComboBox()
            w.addItems(spec.choices or [])
            w.setCurrentText(str(val))
            w.currentTextChanged.connect(lambda text, k=spec.key: self._emit(k, text))
        else:
            w = _SliderRow(spec)
            w.set_value(val)
            w.valueChanged.connect(lambda k=spec.key, ww=w: self._emit(k, ww.value()))
        return w

    def _emit(self, key: str, val: Any) -> None:
        if not self._loading:
            self.valueChanged.emit(key, val)

    def get_values(self) -> dict[str, Any]:
        out = {}
        for key, (spec, w) in self._widgets.items():
            if spec.kind == "bool":
                out[key] = w.isChecked()
            elif spec.kind == "combo":
                out[key] = w.currentText()
            else:
                out[key] = w.value()
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        self._loading = True
        for key, val in values.items():
            if key not in self._widgets:
                continue
            _, w = self._widgets[key]
            if isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(val))
            else:
                w.set_value(val)
        self._loading = False
