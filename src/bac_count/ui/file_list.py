"""左侧文件列表：图片清单、拖放导入、删除条目。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QListWidget, QListWidgetItem

from bac_count.core.preprocess import IMG_EXTS


def extract_image_paths(urls) -> list[Path]:
    """从拖放的 QUrl 提取图片文件；文件夹自动展开；保持传入顺序。"""
    out: list[Path] = []
    for u in urls:
        p = Path(u.toLocalFile())
        if p.is_dir():
            out += sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS)
        elif p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


class FileList(QListWidget):
    pathsDropped = Signal(list)      # list[Path]
    deleteRequested = Signal(list)   # list[Path]
    clearRequested = Signal()
    copyParamsToSelected = Signal(list)  # 当前图参数 → 选中项 list[Path]
    copyParamsToAll = Signal()
    itemPicked = Signal(int)         # 无修饰单击/双击/Enter → 切换图片（行号）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setUniformItemSizes(True)
        self.setToolTip("拖入图片/文件夹追加导入；单击切换图片；Ctrl/Shift 多选（不切换）；"
                        "Delete 或右键移除选中；右键可复制当前图参数到选中")

    # ---------- 同步清单（仅结构变化时重建，避免销毁选区） ----------
    def set_paths(self, paths: list[Path]) -> None:
        self.blockSignals(True)
        self.clear()
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.addItem(item)
        self.blockSignals(False)

    def mark_current(self, current: int) -> None:
        if 0 <= current < self.count():
            self.blockSignals(True)
            if self.currentRow() != current:
                self.setCurrentRow(current)
            self.scrollToItem(self.item(current))
            self.blockSignals(False)

    def selected_paths(self, all_paths: list[Path]) -> list[Path]:
        return [all_paths[i.row()] for i in self.selectedItems() if 0 <= i.row() < len(all_paths)]

    # ---------- 导航手势：仅无修饰左键 / 双击 / Enter ----------
    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and ev.modifiers() == Qt.NoModifier:
            it = self.itemAt(ev.position().toPoint())
            if it is not None:
                super().mouseReleaseEvent(ev)
                self.itemPicked.emit(self.row(it))
                return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        super().mouseDoubleClickEvent(ev)
        if ev.button() == Qt.LeftButton:
            it = self.itemAt(ev.position().toPoint())
            if it is not None:
                self.itemPicked.emit(self.row(it))

    # ---------- 拖放 ----------
    def _mime_ok(self, ev) -> bool:
        m = ev.mimeData()
        return bool(m.hasUrls()) and bool(extract_image_paths(m.urls()))

    def dragEnterEvent(self, ev) -> None:
        if self._mime_ok(ev):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev) -> None:
        if self._mime_ok(ev):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev) -> None:
        paths = extract_image_paths(ev.mimeData().urls())
        if paths:
            ev.acceptProposedAction()
            self.pathsDropped.emit(paths)
            return
        super().dropEvent(ev)

    # ---------- 删除 / Enter 导航 ----------
    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_Delete and self.selectedItems():
            self.deleteRequested.emit(self.selectedPaths())
            return
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and 0 <= self.currentRow() < self.count():
            self.itemPicked.emit(self.currentRow())
            return
        super().keyPressEvent(ev)

    def selectedPaths(self) -> list[Path]:
        return [Path(self.item(i.row()).toolTip()) for i in self.selectedItems()]

    def contextMenuEvent(self, ev) -> None:
        menu = QMenu(self)
        act_del = QAction("移除选中", menu)
        act_clear = QAction("清空列表", menu)
        act_del.triggered.connect(lambda: self.deleteRequested.emit(self.selectedPaths()))
        act_clear.triggered.connect(self.clearRequested.emit)
        menu.addAction(act_del)
        menu.addAction(act_clear)
        menu.addSeparator()
        act_cp_sel = QAction("复制当前图参数 → 选中项", menu)
        act_cp_all = QAction("复制当前图参数 → 全部其他图", menu)
        act_cp_sel.triggered.connect(lambda: self.copyParamsToSelected.emit(self.selectedPaths()))
        act_cp_all.triggered.connect(self.copyParamsToAll.emit)
        menu.addAction(act_cp_sel)
        menu.addAction(act_cp_all)
        menu.exec(ev.globalPos())
