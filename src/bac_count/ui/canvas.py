"""预览画布：缩放/平移、菌落叠加、增删交互。"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QToolButton,
)

from bac_count.core.models import Colony

MODE_BROWSE, MODE_ADD, MODE_REMOVE = "browse", "add", "remove"
VIEW_ORIG, VIEW_PREP, VIEW_MARK = "orig", "prep", "mark"

PEN_AUTO = QColor(0, 230, 0)
PEN_MANUAL = QColor(0, 160, 255)
PEN_HOVER = QColor(255, 60, 60)


def bgra_to_qpixmap(bgr: np.ndarray, alpha: np.ndarray | None) -> QPixmap:
    import cv2

    if alpha is None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    else:
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        rgba[..., 3] = alpha
        img = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


class Canvas(QGraphicsView):
    emptyClicked = Signal(float, float)     # 场景坐标
    colonyClicked = Signal(object)          # Colony
    modeHintChanged = Signal(str)           # 状态栏提示（临时模式切换反馈）
    pathsDropped = Signal(list)             # 拖入图片 list[Path]
    pickPicked = Signal(int, int, int)      # 取色模式：RGB
    pickCancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor(38, 40, 44))
        self._pix = self._scene.addPixmap(bgra_to_qpixmap(np.zeros((10, 10, 3), np.uint8), None))
        self._prep_pix = self._scene.addPixmap(QPixmap())
        self._prep_pix.setZValue(5)
        self._prep_pix.hide()
        self._overlay: list = []
        self._colonies: list[Colony] = []
        self._scale = 1.0  # 场景(工作图) = 原图 * scale
        self._mode = MODE_BROWSE
        self._view_mode = VIEW_MARK
        self._hover: Colony | None = None
        self._panning = False
        self._pan_start = QPoint()
        self._space_pan = False   # 空格临时平移
        self._ctrl_held = False   # Ctrl 临时反转 添加/删除
        self._picking = False     # 取色模式（panel 🎨 触发）
        self._build_view_bar()

    # ---------- 左上角视图切换条 ----------
    def _build_view_bar(self) -> None:
        bar = QFrame(self)
        bar.setObjectName("viewBar")
        bar.setStyleSheet(
            "#viewBar{background:rgba(40,42,46,200); border-radius:6px;}"
            "QToolButton{color:#ddd; background:transparent; border:none; padding:4px 10px; border-radius:4px;}"
            "QToolButton:checked{background:#3d6b99; color:white;}"
            "QToolButton:hover{background:rgba(255,255,255,40);}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self._view_btns: dict[str, QToolButton] = {}
        for label, mode in (("原图", VIEW_ORIG), ("预处理", VIEW_PREP), ("标记", VIEW_MARK)):
            b = QToolButton(bar)
            b.setText(label)
            b.setCheckable(True)
            b.setChecked(mode == self._view_mode)
            b.clicked.connect(lambda _=False, m=mode: self.set_view_mode(m))
            lay.addWidget(b)
            self._view_btns[mode] = b
        bar.adjustSize()
        bar.move(6, 6)
        bar.raise_()
        self._view_bar = bar

    # ---------- 图像与标注 ----------
    def set_image(self, pixmap) -> None:
        self._pix.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fit_view()
        self._apply_view()

    def set_prep_image(self, pixmap) -> None:
        """检测线程回传的预处理图（工作尺度）；None 表示清除。"""
        if pixmap is None:
            self._prep_pix.setPixmap(QPixmap())
        else:
            self._prep_pix.setPixmap(pixmap)
        self._apply_view()

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        for m, b in self._view_btns.items():
            b.blockSignals(True)
            b.setChecked(m == mode)
            b.blockSignals(False)
        self._apply_view()

    def _apply_view(self) -> None:
        has_prep = not self._prep_pix.pixmap().isNull()
        prep_on = self._view_mode == VIEW_PREP and has_prep
        self._prep_pix.setVisible(prep_on)
        self._pix.setVisible(self._view_mode != VIEW_PREP or not has_prep)
        for item in self._overlay:
            item.setVisible(self._view_mode == VIEW_MARK)

    def clear_image(self) -> None:
        self.set_image(bgra_to_qpixmap(np.zeros((10, 10, 3), np.uint8), None))
        self.set_prep_image(None)

    def set_colonies(self, colonies: Sequence[Colony], scale: float) -> None:
        """colonies 为原图坐标；scale 换算到场景。"""
        self._colonies = list(colonies)
        self._scale = scale
        self._redraw_overlay()

    def _redraw_overlay(self) -> None:
        for item in self._overlay:
            self._scene.removeItem(item)
        self._overlay.clear()
        s = self._scale
        font = QFont()
        font.setPointSizeF(8.0)
        for c in self._colonies:
            pen = QPen(PEN_HOVER if c is self._hover else (PEN_MANUAL if c.source == "manual" else PEN_AUTO))
            pen.setWidthF(2.0 if c is self._hover else 1.2)
            r = max(c.r * s, 3.0)
            el = QGraphicsEllipseItem(QRectF(c.x * s - r, c.y * s - r, 2 * r, 2 * r))
            el.setPen(pen)
            el.setZValue(10)
            el.setVisible(self._view_mode == VIEW_MARK)
            self._scene.addItem(el)
            self._overlay.append(el)
            txt = QGraphicsSimpleTextItem(str(c.id))
            txt.setFont(font)
            txt.setBrush(pen.color())
            txt.setPos(c.x * s + r + 1, c.y * s - r - 1)
            txt.setZValue(11)
            txt.setVisible(self._view_mode == VIEW_MARK)
            self._scene.addItem(txt)
            self._overlay.append(txt)

    # ---------- 模式 ----------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if not self._picking:
            self._apply_drag_and_cursor()

    # ---------- 取色模式 ----------
    def enter_pick(self) -> None:
        self._picking = True
        self.setDragMode(QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.CrossCursor)
        self.modeHintChanged.emit("🎨 取色：左键点击颜色（Ctrl+左键可连取多个），右键/Esc 取消")

    def cancel_pick(self) -> None:
        if not self._picking:
            return
        self._picking = False
        self._apply_drag_and_cursor()
        self.pickCancelled.emit()

    def sample_at(self, sx: float, sy: float) -> tuple[int, int, int] | None:
        """从**原图** pixmap（非预处理视图）按场景坐标取 RGB；皿外透明像素返回 None。"""
        pm = self._pix.pixmap()
        if pm.isNull():
            return None
        img = pm.toImage()
        x = min(max(int(round(sx)), 0), img.width() - 1)
        y = min(max(int(round(sy)), 0), img.height() - 1)
        col = img.pixelColor(x, y)
        if col.alpha() < 128:
            return None
        return (col.red(), col.green(), col.blue())

    def _effective_mode(self) -> str:
        """按住 Ctrl 时 添加↔删除 临时互换；浏览模式不受影响。"""
        if self._ctrl_held:
            if self._mode == MODE_ADD:
                return MODE_REMOVE
            if self._mode == MODE_REMOVE:
                return MODE_ADD
        return self._mode

    def _apply_drag_and_cursor(self) -> None:
        if self._mode == MODE_BROWSE:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ClosedHandCursor if self._space_pan else Qt.OpenHandCursor)
            self.modeHintChanged.emit("浏览模式：拖拽平移 · 滚轮缩放 · 0 适应窗口（空格=临时平移）")
            return
        if self._space_pan:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        self.setDragMode(QGraphicsView.NoDrag)
        eff = self._effective_mode()
        self.viewport().setCursor(Qt.PointingHandCursor if eff == MODE_REMOVE else Qt.CrossCursor)
        hint = {
            MODE_ADD: "添加模式：点击空白处添加菌落（按住 Ctrl = 临时删除）",
            MODE_REMOVE: "删除模式：点击菌落删除（按住 Ctrl = 临时添加）",
        }.get(self._mode, "")
        if self._ctrl_held and self._mode in (MODE_ADD, MODE_REMOVE):
            hint = f"⚡临时切换为「{'删除' if eff == MODE_REMOVE else '添加'}」：{hint.split('：')[1]}"
        self.modeHintChanged.emit(hint)

    # ---------- 视图 ----------
    def fit_view(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._view_bar.move(6, 6)
        self._view_bar.raise_()

    # ---------- 交互 ----------
    def keyPressEvent(self, ev) -> None:
        if self._picking and ev.key() == Qt.Key_Escape:
            self.cancel_pick()
            ev.accept()
            return
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            if not self._space_pan:
                self._space_pan = True
                self._apply_drag_and_cursor()
            ev.accept()
            return
        if ev.key() == Qt.Key_Control and not ev.isAutoRepeat():
            if not self._ctrl_held:
                self._ctrl_held = True
                self._apply_drag_and_cursor()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev) -> None:
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            self._space_pan = False
            self._apply_drag_and_cursor()
            ev.accept()
            return
        if ev.key() == Qt.Key_Control and not ev.isAutoRepeat():
            self._ctrl_held = False
            self._apply_drag_and_cursor()
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev) -> None:
        # 失焦时按键 release 可能收不到，重置临时状态
        self._space_pan = False
        self._ctrl_held = False
        self._apply_drag_and_cursor()
        super().focusOutEvent(ev)

    def wheelEvent(self, ev) -> None:
        self.zoom(1.15 if ev.angleDelta().y() > 0 else 1 / 1.15)

    def mousePressEvent(self, ev) -> None:
        if self._picking:
            if ev.button() == Qt.LeftButton:
                sp = self.mapToScene(ev.position().toPoint())
                rgb = self.sample_at(sp.x(), sp.y())
                if rgb is not None:
                    self.pickPicked.emit(*rgb)
                    if not (ev.modifiers() & Qt.ControlModifier):  # Ctrl+左键连取
                        self._picking = False
                        self._apply_drag_and_cursor()
                ev.accept()
            elif ev.button() == Qt.RightButton:
                self.cancel_pick()
                ev.accept()
            else:
                super().mousePressEvent(ev)
            return
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.position().toPoint()
            ev.accept()
            return
        eff = self._effective_mode()
        if self._space_pan or eff == MODE_BROWSE:
            super().mousePressEvent(ev)  # ScrollHandDrag 平移
            return
        if ev.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            hit = self._hit(scene_pos.x(), scene_pos.y())
            if eff == MODE_REMOVE and hit is not None:
                self.colonyClicked.emit(hit)
            elif eff == MODE_ADD and hit is None:
                self.emptyClicked.emit(scene_pos.x(), scene_pos.y())
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._panning:
            p = ev.position().toPoint()
            d = p - self._pan_start
            self._pan_start = p
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            ev.accept()
            return
        if self._picking:
            sp = self.mapToScene(ev.position().toPoint())
            rgb = self.sample_at(sp.x(), sp.y())
            self.setToolTip(f"rgb{rgb}（左键取色）" if rgb else "透明像素，不可取色")
            ev.accept()
            return
        super().mouseMoveEvent(ev)
        scene_pos = self.mapToScene(ev.position().toPoint())
        hit = self._hit(scene_pos.x(), scene_pos.y())
        if hit is not self._hover:
            self._hover = hit
            self._redraw_overlay()
        if hit is not None:
            self.setToolTip(f"#{hit.id} ({hit.x:.0f},{hit.y:.0f}) r={hit.r:.1f} {hit.source}")
        else:
            self.setToolTip("")

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MiddleButton:
            self._panning = False
        super().mouseReleaseEvent(ev)

    def _hit(self, sx: float, sy: float) -> Colony | None:
        best, best_d = None, None
        for c in self._colonies:
            d = ((c.x * self._scale - sx) ** 2 + (c.y * self._scale - sy) ** 2) ** 0.5
            if d <= max(c.r * self._scale + 4, 10) and (best_d is None or d < best_d):
                best, best_d = c, d
        return best

    # ---------- 拖放导入 ----------
    def dragEnterEvent(self, ev) -> None:
        from bac_count.ui.file_list import extract_image_paths

        if ev.mimeData().hasUrls() and extract_image_paths(ev.mimeData().urls()):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()  # dragEnter 已验证过图片类型
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev) -> None:
        from bac_count.ui.file_list import extract_image_paths

        paths = extract_image_paths(ev.mimeData().urls())
        if paths:
            ev.acceptProposedAction()
            self.pathsDropped.emit(paths)
        else:
            super().dropEvent(ev)
