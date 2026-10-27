"""汇总预览窗口：把「汇总表 CSV」的内容先在表格里看一遍，看对了再一键导出。

表格内容 = `core.export.write_summary_csv` 写出去的内容（同一份表头/列顺序、
同一批 rows dict），预览完导出时不再重算检测，所以「表里看到的数」和
「CSV 里的数」必然一致——不会预览一套、导出另一套。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from bac_count.core.export import SUMMARY_HEADERS, SUMMARY_KEYS, write_summary_csv

# 第一列额外挂这行在原始 rows 里的下标。点表头排序搬的是**整行**，data 跟着走，
# 于是导出时按当前显示顺序还原原始 dict：表格所见即所导出（包括排过序之后）。
_ROW_ID = Qt.UserRole + 1
_NUMERIC_KEYS = {"count", "auto_total", "added", "removed"}


class SummaryPreview(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("汇总预览")
        self.resize(760, 480)
        self.setModal(False)  # 非模态：一边看表一边继续调参/加减菌落，再点一次预览刷新

        self._rows: list[dict] = []
        self._default_dir = Path.cwd()
        self._sort: tuple[int, Qt.SortOrder] | None = None  # 用户选过的排序，刷新后照旧

        # ---------- 表格 ----------
        self.model = QStandardItemModel(self)
        self.table = QTableView(self)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 预览表不是编辑器
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)  # 数字列存的是数字，按大小排而不是按字符排
        hh = self.table.horizontalHeader()
        hh.setSortIndicatorShown(True)
        hh.sortIndicatorChanged.connect(self._remember_sort)  # 记下用户选的排序，刷新时保住它
        hh.setSectionResizeMode(0, QHeaderView.Stretch)  # 文件名最长 → 吃满剩余宽度
        for col in range(1, len(SUMMARY_HEADERS)):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setToolTip("内容与「📤 导出 → 汇总表 CSV」完全一致。\n"
                              "默认行序 = 文件列表顺序（= 批量导出的行序）；点表头可排序，"
                              "排序后「导出 CSV」就按当前显示顺序写。\nCtrl+C 复制选中行，可直接粘进表格软件。")

        # ---------- 表格下方：统计 + 导出 ----------
        self._lbl_info = QLabel("没有数据")
        self._lbl_fail = QLabel("")
        self._lbl_fail.setVisible(False)
        self._lbl_fail.setStyleSheet("color:#b00020;")
        self._btn_export = QPushButton("💾 导出 CSV…")
        self._btn_export.clicked.connect(self.export_csv)
        self._btn_export.setToolTip("把当前表格内容写成 CSV（不重新检测，与预览一致）")
        self._btn_export.setEnabled(False)
        self._btn_close = QPushButton("关闭")
        self._btn_close.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(self._lbl_info)
        row.addStretch(1)
        row.addWidget(self._btn_export)
        row.addWidget(self._btn_close)

        lay = QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        lay.addLayout(row)
        lay.addWidget(self._lbl_fail)

    # ------------------------------------------------------------------ 填表
    def _remember_sort(self, col: int, order) -> None:
        self._sort = (col, order)

    def set_rows(self, rows: list[dict], failures: list[str] = (), default_dir=None) -> None:
        self._rows = list(rows)
        if default_dir:
            self._default_dir = Path(default_dir)

        self.table.setSortingEnabled(False)  # 填充期间关排序：否则每 append 一次就重排一次
        self.model.clear()  # clear 连表头一起清（实测 headerData 变 None）→ 之后必须重设
        self.model.setHorizontalHeaderLabels(SUMMARY_HEADERS)
        for i, r in enumerate(self._rows):
            items = []
            for col, key in enumerate(SUMMARY_KEYS):
                it = QStandardItem()
                if key in _NUMERIC_KEYS:
                    it.setData(int(r[key]), Qt.DisplayRole)  # 存 int 才会 9 < 10（存文本会排成 10,100,9）
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    it.setData(str(r[key]), Qt.DisplayRole)
                it.setEditable(False)
                if col == 0:
                    it.setData(i, _ROW_ID)  # 排序后还能认得出原来是哪一行
                items.append(it)
            self.model.appendRow(items)
        self.table.setSortingEnabled(True)
        if self._sort is not None:
            # 重放用户选的排序。不重放就会这样：行序悄悄回到文件列表序，表头箭头却还挂着 →
            # 这时点「导出 CSV」写出去的是文件序 ≠ 屏幕上看到的序，正是这个窗口要避免的错配。
            self.table.sortByColumn(*self._sort)

        total = sum(int(r["count"]) for r in self._rows)
        self._lbl_info.setText(f"共 {len(self._rows)} 行 · 总菌落 {total} 个")
        self._btn_export.setEnabled(bool(self._rows))

        if failures:
            detail = "\n".join(failures[:8])
            if len(failures) > 8:
                detail += f"\n…（另有 {len(failures) - 8} 条）"
            self._lbl_fail.setText(f"⚠ {len(failures)} 张图片汇总失败，未列入表格")
            self._lbl_fail.setToolTip(detail)
            self._lbl_fail.setVisible(True)
        else:
            self._lbl_fail.clear()
            self._lbl_fail.setToolTip("")
            self._lbl_fail.setVisible(False)

    # ------------------------------------------------------------------ 导出
    def _display_rows(self) -> list[dict]:
        """按**当前显示顺序**取原始 rows（排过序就跟着排后的顺序走）。"""
        out: list[dict] = []
        for row in range(self.model.rowCount()):
            i = self.model.index(row, 0).data(_ROW_ID)
            if isinstance(i, int) and 0 <= i < len(self._rows):
                out.append(self._rows[i])
        return out

    def export_csv(self) -> None:
        rows = self._display_rows()
        if not rows:
            return
        f, _ = QFileDialog.getSaveFileName(self, "导出汇总 CSV",
                                           str(self._default_dir / "counts_summary.csv"), "CSV (*.csv)")
        if not f:
            return
        try:
            write_summary_csv(f, rows)
        except Exception as e:  # 写盘失败（磁盘满/被占/非法字符）不能静默
            QMessageBox.critical(self, "写入失败", f"{f}\n\n{e}")
            return
        self._lbl_info.setText(f"已导出 {len(rows)} 行 → {f}")
        bar = self.parent().statusBar() if hasattr(self.parent(), "statusBar") else None
        if bar is not None:
            bar.showMessage(f"已导出汇总 CSV（{len(rows)} 行）到 {f}", 6000)
