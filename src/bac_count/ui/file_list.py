"""左侧文件列表：图片清单、拖放导入、删除条目。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
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
        self._nav_row = -1          # 正在显示的图片行（加粗标记）
        self.current_path = ""      # 正在显示的图片路径（复制源）
        self.setToolTip("拖入图片/文件夹追加导入；单击切换图片；Ctrl/Shift 多选（不切换）；"
                        "Delete 或右键移除选中；右键可复制当前图参数到选中")

    # ---------- 同步清单（仅结构变化时重建，避免销毁选区） ----------
    def set_paths(self, paths: list[Path]) -> None:
        self.blockSignals(True)
        self.clear()
        self._nav_row = -1
        self.current_path = ""
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.addItem(item)
        self.blockSignals(False)

    def mark_current(self, current: int) -> None:
        """标记正在显示的行（加粗）。只动标记，不销毁选区。"""
        if not (0 <= current < self.count()) or current == self._nav_row:
            return
        self.blockSignals(True)
        if 0 <= self._nav_row < self.count():
            it = self.item(self._nav_row)
            f = it.font()
            f.setBold(False)
            it.setFont(f)
        it = self.item(current)
        f = it.font()
        f.setBold(True)
        it.setFont(f)
        self._nav_row = current
        self.current_path = it.toolTip()
        self.scrollToItem(it)
        if self.currentRow() != current:
            self.setCurrentRow(current)
        self.blockSignals(False)

    # ---------- 导航手势：仅无修饰左键 / 双击 / Enter ----------
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.RightButton:
            it = self.itemAt(ev.position().toPoint())
            if it is not None and not it.isSelected():
                # 默认行为会清空多选只留右键项 → 改为加选，保住已有选区
                it.setSelected(True)
                ev.accept()
                return
        super().mousePressEvent(ev)

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
        # 注意：QListWidgetItem.row() 在 PySide6 不暴露，必须用 QListWidget.row(item)
        return [Path(self.item(self.row(i)).toolTip()) for i in self.selectedItems()]

    def contextMenuEvent(self, ev) -> None:
        # 打开瞬间捕获选区快照：菜单存续期间任何状态漂移都不影响执行目标
        snapshot = self.selectedPaths()
        n_targets = sum(1 for p in snapshot if str(p) != self.current_path)
        menu = QMenu(self)
        act_del = menu.addAction(f"移除选中 ({len(snapshot)})")
        act_del.setEnabled(bool(snapshot))
        act_del.triggered.connect(lambda: self.deleteRequested.emit(list(snapshot)))
        act_clear = menu.addAction("清空列表")
        act_clear.triggered.connect(self.clearRequested.emit)
        menu.addSeparator()
        act_cp_sel = menu.addAction(f"复制当前图参数 → 选中项 ({n_targets})")
        act_cp_sel.setEnabled(n_targets > 0)
        act_cp_sel.setToolTip("源 = 加粗的当前图；目标 = 选中的其他图片")
        act_cp_sel.triggered.connect(lambda: self.copyParamsToSelected.emit(list(snapshot)))
        act_cp_all = menu.addAction("复制当前图参数 → 全部其他图")
        act_cp_all.triggered.connect(self.copyParamsToAll.emit)
        self._popup_menu(menu, ev.globalPos())  # PySide6 未暴露 globalPosition()

    def _popup_menu(self, menu: QMenu, pos) -> None:
        menu.exec(pos)
