"""左侧文件列表：图片清单、拖放导入、拖动排序、删除条目。"""
from __future__ import annotations

import re
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


def natural_sort_key(text) -> list[tuple]:
    """自然排序键：名字里的数字按**数值**比大小，所以 图2 排在 图10 前面。

    'E0-3' < 'E12-2'；'plate9.png' < 'plate10.png'；大小写不敏感。
    每段编成 (类型, 数值, 文本) 三元组，避免 int 与 str 直接比较报错。
    """
    parts = [p for p in re.split(r"(\d+)", str(text)) if p != ""]
    return [(0, int(p), "") if p.isdigit() else (1, 0, p.lower()) for p in parts]


class FileList(QListWidget):
    pathsDropped = Signal(list)      # list[Path]
    deleteRequested = Signal(list)   # list[Path]
    clearRequested = Signal()
    copyParamsToSelected = Signal(list)  # 当前图参数 → 选中项 list[Path]
    copyParamsToAll = Signal()
    itemPicked = Signal(int)         # 无修饰单击/双击/Enter → 切换图片（行号）
    orderChanged = Signal(list)      # 拖动排序/按名称排序 → 新顺序 list[Path]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setUniformItemSizes(True)
        # 拖动排序：InternalMove 让 Qt 自己搬条目（带插入位置指示线、多选可整块搬）；
        # 外部图片/文件夹拖入不受影响 —— 内部拖动的 mime 是
        # 'application/x-qabstractitemmodeldatalist'，hasUrls() 为 False，走不到 _mime_ok 那条路。
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDropIndicatorShown(True)
        self._nav_row = -1          # 正在显示的图片行（加粗标记）
        self.current_path = ""      # 正在显示的图片路径（复制源）
        self._dragging = False      # 拖动进行中/刚结束 → 抑制随后的"单击切图"
        self.setToolTip("拖入图片/文件夹追加导入；拖动条目可改变顺序（=导航与导出顺序）；"
                        "单击切换图片；Ctrl/Shift 多选（不切换）；Delete 或右键移除选中；"
                        "右键可复制当前图参数到选中、按名称排序")

    # ---------- 顺序 ----------
    def pathsInOrder(self) -> list[Path]:
        return [Path(self.item(i).toolTip()) for i in range(self.count())]

    def _refresh_nav_row(self) -> None:
        """条目被搬动后行号全变了：加粗标记与 current_path 按**路径**重新定位。"""
        row = -1
        if self.current_path:
            for i in range(self.count()):
                if self.item(i).toolTip() == self.current_path:
                    row = i
                    break
        self._nav_row = row
        if row >= 0 and self.currentRow() != row:
            # setCurrentIndex 只挪 current（键盘 Enter/方向键用的那一行），不动选区
            self.setCurrentIndex(self.model().index(row, 0))

    def _reselect(self, tooltips: set) -> None:
        """按**文件**重建选区。必须等顺序与 current 都落定之后再调用：
        中途任何 setCurrentRow/setCurrentIndex 都会把多选塌成单选。"""
        if not tooltips:
            return
        self.clearSelection()
        for i in range(self.count()):
            if self.item(i).toolTip() in tooltips:
                self.item(i).setSelected(True)

    def _order_changed(self) -> None:
        self._refresh_nav_row()
        self.orderChanged.emit(self.pathsInOrder())

    def startDrag(self, supportedActions) -> None:
        """Qt 内部移动完成后，把新顺序广播给主窗口；并吃掉紧接着的那次 release。"""
        before = self.pathsInOrder()
        selected = {it.toolTip() for it in self.selectedItems()}
        self._dragging = True
        try:
            super().startDrag(supportedActions)
        finally:
            after = self.pathsInOrder()
            if after != before:
                self._order_changed()
                self._reselect(selected)  # 选区跟着文件走，不跟着行号走
            # _dragging 留着 True：Qt 还会补一次 release，由 mouseReleaseEvent 消费掉

    def sortByName(self) -> None:
        """按名称排序（数字按大小）。条目整体搬动，加粗/选区跟着**文件**走。"""
        n = self.count()
        if n < 2:
            return
        # 选区必须在取出条目之前记录：条目一旦脱离列表，选中状态就不可靠了
        selected = {it.toolTip() for it in self.selectedItems()}
        items = [self.takeItem(0) for _ in range(n)]  # 全部取出，顺序即原行序
        order = sorted(range(n), key=lambda i: (natural_sort_key(items[i].text()),
                                                natural_sort_key(items[i].toolTip())))
        self.blockSignals(True)
        for pos, src in enumerate(order):
            self.insertItem(pos, items[src])  # 复用同一批条目对象 → 文本/路径/加粗跟着走
        self.blockSignals(False)
        self._order_changed()
        self._reselect(selected)  # 最后一步：行号与 current 都定了再重建选区

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
        self._dragging = False  # 新的按压：上一次拖动作废
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
            was_drag = self._dragging
            self._dragging = False
            if it is not None and not was_drag:
                # 拖完序 Qt 还会送一次 release：不吞掉就会"排完序结果切到别的图"
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
        before = self.pathsInOrder()
        super().dropEvent(ev)  # 内部拖动排序由 Qt 搬条目
        if self.pathsInOrder() != before:
            self._order_changed()  # 兜底：任何走 model 的移动都同步出去

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
        act_sort = menu.addAction("按名称排序")
        act_sort.setEnabled(self.count() > 1)
        act_sort.setToolTip("数字按大小比较：图2 排在 图10 前面；当前显示的图片跟着自己走")
        act_sort.triggered.connect(self.sortByName)
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
