"""预览画布：缩放/平移、菌落叠加、增删交互。"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QToolButton,
    QWidget,
)

from bac_count.core.models import Colony

MODE_BROWSE, MODE_ADD, MODE_REMOVE = "browse", "add", "remove"
VIEW_ORIG, VIEW_PREP = "orig", "prep"

PEN_AUTO = QColor(0, 230, 0)
PEN_MANUAL = QColor(0, 160, 255)
PEN_HOVER = QColor(255, 60, 60)


class _LoupeWidget(QWidget):
    """取色放大镜：圆形局部放大 + 中心十字 + 颜色样本条（鼠标旁，不拦事件）。"""

    D = 132          # 圆直径 px
    SRC = 25         # 默认源区域（场景像素）；实际按视图缩放动态算
    H = 160

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.D + 8, self.H)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._img: QImage | None = None
        self._cx = self._cy = 0
        self._rgb: tuple | None = None
        self._src = float(self.SRC)

    def set_sample(self, img: QImage | None, cx: int, cy: int, rgb, src: float | None = None) -> None:
        self._img, self._cx, self._cy, self._rgb = img, cx, cy, rgb
        if src is not None:
            self._src = src
        self.update()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        ring = QRectF(4, 4, self.D - 4, self.D - 4)
        path = QPainterPath()
        path.addEllipse(ring)
        if self._img is not None and not self._img.isNull():
            p.setClipPath(path)
            half = self._src / 2
            src = QRectF(self._cx + 0.5 - half, self._cy + 0.5 - half, self._src, self._src)
            p.drawImage(ring, self._img, src)   # 最近邻缩放，看得清单像素
            p.setClipping(False)
        else:
            p.fillPath(path, QColor(70, 70, 70))
        p.setPen(QPen(QColor(255, 255, 255, 230), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(ring)
        ctr = ring.center()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):   # 中心十字
            p.setPen(QPen(QColor(0, 0, 0, 160), 3))
            p.drawLine(ctr.x() + dx * 3, ctr.y() + dy * 3, ctr.x() + dx * 11, ctr.y() + dy * 11)
            p.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
            p.drawLine(ctr.x() + dx * 3, ctr.y() + dy * 3, ctr.x() + dx * 11, ctr.y() + dy * 11)
        chip = QRectF(4, self.D + 2, self.D - 4, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(40, 42, 46, 225))
        p.drawRoundedRect(chip, 5, 5)
        if self._rgb is not None:
            p.setBrush(QColor(*self._rgb))
            p.drawRoundedRect(QRectF(8, self.D + 5, 16, 16), 3, 3)
            p.setPen(QColor(230, 230, 230))
            p.drawText(QRectF(28, self.D + 2, self.D - 30, 22), Qt.AlignVCenter,
                       f"{self._rgb[0]},{self._rgb[1]},{self._rgb[2]}")
        else:
            p.setPen(QColor(180, 180, 180))
            p.drawText(chip, Qt.AlignCenter, "透明像素")
        p.end()


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
    LOUPE_REL = 3.0                     # 放大镜相对屏幕当前观感的再放大倍数
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
        self._view_mode = VIEW_ORIG
        self._show_marks = True   # 标记叠加（菌落圈/编号）显隐，独立于视图选择
        self._hover: Colony | None = None
        self._panning = False
        self._pan_start = QPoint()
        self._space_pan = False   # 空格临时平移
        self._ctrl_held = False   # Ctrl 临时反转 添加/删除
        self._picking = False     # 取色模式（panel 🎨 触发）
        self._pick_img: QImage | None = None   # 取色会话的原图缓存
        self._pick_pos: QPointF | None = None  # 最近一次取色鼠标视口坐标（滚轮后刷新放大镜）
        self._build_view_bar()
        self._loupe = _LoupeWidget(self.viewport())
        self._loupe.hide()

    # ---------- 左上视图切换条 / 右上标记开关 ----------
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
        for label, mode in (("原图", VIEW_ORIG), ("预处理", VIEW_PREP)):
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

        mbar = QFrame(self)
        mbar.setObjectName("marksBar")
        mbar.setStyleSheet(
            "#marksBar{background:rgba(40,42,46,200); border-radius:6px;}"
            "QCheckBox{color:#ddd; spacing:5px;}"
        )
        play = QHBoxLayout(mbar)
        play.setContentsMargins(8, 4, 8, 4)
        self._chk_marks = QCheckBox("显示标记", mbar)
        self._chk_marks.setChecked(self._show_marks)
        self._chk_marks.setCursor(Qt.PointingHandCursor)
        self._chk_marks.setToolTip("菌落圆圈/编号叠加的显隐（原图与预处理视图通用）")
        self._chk_marks.toggled.connect(self.set_show_marks)
        play.addWidget(self._chk_marks)
        mbar.adjustSize()
        self._marks_bar = mbar
        self._reposition_floats()

    def _reposition_floats(self) -> None:
        self._view_bar.move(6, 6)
        self._marks_bar.move(self.width() - self._marks_bar.width() - 6, 6)
        self._view_bar.raise_()
        self._marks_bar.raise_()

    # ---------- 图像与标注 ----------
    def set_image(self, pixmap) -> None:
        if self._picking:
            self.cancel_pick()  # 换图后旧缓存/坐标语义失效
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
        self._pix.setVisible(not prep_on)  # 无预处理图时回退原图
        for item in self._overlay:
            item.setVisible(self._show_marks)

    def set_show_marks(self, on: bool) -> None:
        self._show_marks = bool(on)
        for item in self._overlay:
            item.setVisible(self._show_marks)

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
            el.setVisible(self._show_marks)
            self._scene.addItem(el)
            self._overlay.append(el)
            txt = QGraphicsSimpleTextItem(str(c.id))
            txt.setFont(font)
            txt.setBrush(pen.color())
            txt.setPos(c.x * s + r + 1, c.y * s - r - 1)
            txt.setZValue(11)
            txt.setVisible(self._show_marks)
            self._scene.addItem(txt)
            self._overlay.append(txt)

    # ---------- 模式 ----------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if not self._picking:
            self._apply_drag_and_cursor()

    # ---------- 取色模式 ----------
    def enter_pick(self) -> None:
        pm = self._pix.pixmap()
        self._pick_img = None if pm.isNull() else pm.toImage()
        self._pick_pos = None
        self._picking = True
        self.setFocus()  # 🎨 按钮点击会把焦点留在面板，空格/Ctrl/Esc 必须直达画布
        self._apply_drag_and_cursor()
        self._loupe.show()
        self._loupe.raise_()

    def _end_pick(self) -> None:
        self._picking = False
        self._pick_img = None
        self._pick_pos = None
        self._loupe.hide()
        self._apply_drag_and_cursor()

    def cancel_pick(self) -> None:
        if not self._picking:
            return
        self._end_pick()
        self.pickCancelled.emit()

    def sample_at(self, sx: float, sy: float) -> tuple[int, int, int] | None:
        """从**原图**（非预处理视图）按场景坐标取 RGB；皿外透明像素返回 None。"""
        img = self._pick_img if self._pick_img is not None else None
        if img is None:
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

    def _update_loupe(self, scene_pos: QPointF, vp_pos) -> None:
        ix = int(round(scene_pos.x()))
        iy = int(round(scene_pos.y()))
        rgb = self.sample_at(scene_pos.x(), scene_pos.y())
        # 源区域随视图缩放自适应：放大镜内观感 = 屏幕当前观感 × LOUPE_REL
        s = abs(self.transform().m11()) or 1.0
        src = max(4.0, min(160.0, _LoupeWidget.D / (s * self.LOUPE_REL)))
        self._loupe.set_sample(self._pick_img, ix, iy, rgb, src)
        if not self._loupe.isVisible():
            self._loupe.show()
            self._loupe.raise_()
        vw, vh = self.viewport().width(), self.viewport().height()
        x = int(vp_pos.x()) + 12
        y = int(vp_pos.y()) + 10
        if x + self._loupe.width() > vw:
            x = int(vp_pos.x()) - 12 - self._loupe.width()
        if y + self._loupe.height() > vh:
            y = int(vp_pos.y()) - 10 - self._loupe.height()
        self._loupe.move(max(0, x), max(0, y))

    def _effective_mode(self) -> str:
        """按住 Ctrl 时 添加↔删除 临时互换；浏览模式不受影响。"""
        if self._ctrl_held:
            if self._mode == MODE_ADD:
                return MODE_REMOVE
            if self._mode == MODE_REMOVE:
                return MODE_ADD
        return self._mode

    def _apply_drag_and_cursor(self) -> None:
        if self._picking:
            if self._space_pan:  # 取色中按住空格 → 临时平移，放大镜让位
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self.viewport().setCursor(Qt.ClosedHandCursor)
                self._loupe.hide()
                self.modeHintChanged.emit("🎨 取色暂停：空格平移中，松开回到取色")
            else:
                self.setDragMode(QGraphicsView.NoDrag)
                self.viewport().setCursor(Qt.CrossCursor)
                self._loupe.show()
                self._loupe.raise_()
                self.modeHintChanged.emit(
                    "🎨 取色：左键点击颜色（Ctrl+左键连取，按住空格临时平移），右键/Esc 取消")
            return
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
        self._reposition_floats()

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

    def leaveEvent(self, ev) -> None:
        if self._picking:
            self._loupe.hide()
        super().leaveEvent(ev)

    def wheelEvent(self, ev) -> None:
        self.zoom(1.15 if ev.angleDelta().y() > 0 else 1 / 1.15)
        if self._picking and self._pick_pos is not None:
            # 缩放后放大镜原地刷新（源区域随新倍率变化）
            self._update_loupe(self.mapToScene(self._pick_pos.toPoint()), self._pick_pos)

    def mousePressEvent(self, ev) -> None:
        if self._picking:
            if self._space_pan:  # 空格平移：交给 ScrollHandDrag 处理
                super().mousePressEvent(ev)
                return
            if ev.button() == Qt.LeftButton:
                sp = self.mapToScene(ev.position().toPoint())
                rgb = self.sample_at(sp.x(), sp.y())
                if rgb is not None:
                    self.pickPicked.emit(*rgb)
                    if not (ev.modifiers() & Qt.ControlModifier):  # Ctrl+左键连取
                        self._end_pick()
                    else:
                        self._update_loupe(sp, ev.position())
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
            if self._space_pan:
                super().mouseMoveEvent(ev)  # 平移拖拽由基类处理
                return
            # 必须转给基类：AnchorUnderMouse 依赖基类跟踪的鼠标位置，否则滚轮缩放锚点漂移
            super().mouseMoveEvent(ev)
            sp = self.mapToScene(ev.position().toPoint())
            self._pick_pos = QPointF(ev.position())
            self._update_loupe(sp, ev.position())
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
