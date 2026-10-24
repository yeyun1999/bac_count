"""预览画布：缩放/平移、菌落叠加、增删交互。"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from bac_count.core.models import Colony

MODE_BROWSE, MODE_ADD, MODE_REMOVE = "browse", "add", "remove"

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor(38, 40, 44))
        self._pix = self._scene.addPixmap(bgra_to_qpixmap(np.zeros((10, 10, 3), np.uint8), None))
        self._overlay: list = []
        self._colonies: list[Colony] = []
        self._scale = 1.0  # 场景(工作图) = 原图 * scale
        self._mode = MODE_BROWSE
        self._hover: Colony | None = None
        self._panning = False
        self._pan_start = QPoint()

    # ---------- 图像与标注 ----------
    def set_image(self, pixmap) -> None:
        self._pix.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fit_view()

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
            self._scene.addItem(el)
            self._overlay.append(el)
            txt = QGraphicsSimpleTextItem(str(c.id))
            txt.setFont(font)
            txt.setBrush(pen.color())
            txt.setPos(c.x * s + r + 1, c.y * s - r - 1)
            txt.setZValue(11)
            self._scene.addItem(txt)
            self._overlay.append(txt)

    # ---------- 模式 ----------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == MODE_BROWSE:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    # ---------- 视图 ----------
    def fit_view(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    # ---------- 交互 ----------
    def wheelEvent(self, ev) -> None:
        self.zoom(1.15 if ev.angleDelta().y() > 0 else 1 / 1.15)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.pos()
            ev.accept()
            return
        if ev.button() == Qt.LeftButton and self._mode != MODE_BROWSE:
            scene_pos = self.mapToScene(ev.pos())
            hit = self._hit(scene_pos.x(), scene_pos.y())
            if self._mode == MODE_REMOVE and hit is not None:
                self.colonyClicked.emit(hit)
            elif self._mode == MODE_ADD and hit is None:
                self.emptyClicked.emit(scene_pos.x(), scene_pos.y())
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._panning:
            d = ev.pos() - self._pan_start
            self._pan_start = ev.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            ev.accept()
            return
        super().mouseMoveEvent(ev)
        scene_pos = self.mapToScene(ev.pos())
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
