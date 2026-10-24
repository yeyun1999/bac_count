"""参数面板：schema 驱动控件生成；分组可折叠，折叠状态记忆；标签/控件带悬浮解释。"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bac_count.core.models import ParamSpec

# 首次使用时默认收起的分组（用户手动改过则以记忆为准）
DEFAULT_COLLAPSED = {"预处理", "后处理过滤", "手动校正"}

_SETTINGS_ORG = "bac_count"


def default_settings() -> QSettings:
    return QSettings(_SETTINGS_ORG, _SETTINGS_ORG)


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


class _HeaderWidget(QWidget):
    """可点击的分组标题行（左对齐；QToolButton 会忽略样式表 text-align）。"""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("grpHeader")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "#grpHeader{background:transparent;}"
            "#grpHeader:hover{background:rgba(128,128,128,45); border-radius:4px;}"
            "#grpHeader QLabel{font-weight:600;}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        self.lbl = QLabel(self)
        lay.addWidget(self.lbl)
        lay.addStretch(1)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def click(self) -> None:
        self.clicked.emit()


class CollapsibleBox(QWidget):
    """标题行（▾/▸）+ 可折叠内容区。"""

    collapsedChanged = Signal(str, bool)  # group, collapsed

    def __init__(self, group: str, collapsed: bool, parent=None):
        super().__init__(parent)
        self.group = group
        self._collapsed = collapsed
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = _HeaderWidget(self)
        self._sync_header()
        lay.addWidget(self.header)
        self.content = QWidget(self)
        self.form = QFormLayout(self.content)
        self.form.setContentsMargins(8, 4, 8, 8)
        self.form.setLabelAlignment(Qt.AlignLeft)
        self.content.setVisible(not collapsed)
        lay.addWidget(self.content)
        self.header.clicked.connect(self._toggle)

    def _sync_header(self) -> None:
        self.header.lbl.setText(f"{'▸' if self._collapsed else '▾'}  {self.group}")

    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
        self._sync_header()
        self.collapsedChanged.emit(self.group, self._collapsed)

    def add_row(self, label: QWidget, field: QWidget) -> None:
        self.form.addRow(label, field)


class ParamPanel(QScrollArea):
    valueChanged = Signal(str, object)  # key, value

    def __init__(self, parent=None, settings: QSettings | None = None):
        super().__init__(parent)
        self._settings = settings or default_settings()
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)
        self.setWidget(self._container)
        self._widgets: dict[str, tuple[ParamSpec, QWidget]] = {}
        self._boxes: dict[str, CollapsibleBox] = {}
        self._loading = False

    # ---------- 折叠状态持久化 ----------
    def _collapsed_for(self, group: str) -> bool:
        v = self._settings.value(f"collapsed/{group}", None)
        if v is None:
            return group in DEFAULT_COLLAPSED
        return str(v).lower() in ("true", "1")

    def _remember_collapse(self, group: str, collapsed: bool) -> None:
        self._settings.setValue(f"collapsed/{group}", collapsed)

    # ---------- 构建 ----------
    def set_schema(self, specs: list[ParamSpec], values: dict[str, Any]) -> None:
        """重建控件；values 提供当前值（缺省用 spec.default）。"""
        self._loading = True
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()
        self._boxes.clear()
        for spec in specs:
            if spec.group not in self._boxes:
                box = CollapsibleBox(spec.group, self._collapsed_for(spec.group))
                box.collapsedChanged.connect(self._remember_collapse)
                self._layout.addWidget(box)
                self._boxes[spec.group] = box
            val = values.get(spec.key, spec.default)
            w = self._make_widget(spec, val)
            label = QLabel(spec.label)
            if spec.desc:
                tip = spec.desc
                label.setToolTip(tip)
                w.setToolTip(tip)
            self._boxes[spec.group].add_row(label, w)
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
