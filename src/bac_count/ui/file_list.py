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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setUniformItemSizes(True)
        self.setToolTip("拖入图片或文件夹可追加导入；选中后按 Delete 或右键可移除")

    # ---------- 同步清单 ----------
    def sync(self, paths: list[Path], current: int) -> None:
        self.blockSignals(True)
        self.clear()
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.addItem(item)
        if 0 <= current < self.count():
            self.setCurrentRow(current)
        self.blockSignals(False)

    def selected_paths(self, all_paths: list[Path]) -> list[Path]:
        return [all_paths[i.row()] for i in self.selectedItems() if 0 <= i.row() < len(all_paths)]

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

    # ---------- 删除 ----------
    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_Delete and self.selectedItems():
            self.deleteRequested.emit(self.selectedPaths())
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
