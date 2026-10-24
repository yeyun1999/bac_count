"""主窗口：工具栏 + 左预览画布 + 右参数面板 + 状态栏。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QToolBar,
    QToolButton,
)

from bac_count.core.edits import EditHistory
from bac_count.core.export import export_csv, export_json, export_png
from bac_count.core.models import Colony, merge
from bac_count.core.pipeline import POST_SCHEMA, PRE_SCHEMA, WORKING_MAX, default_params
from bac_count.core.preprocess import IMG_EXTS, load_image
from bac_count.core.registry import DETECTORS, get_detector
from bac_count.ui.canvas import MODE_ADD, MODE_BROWSE, MODE_REMOVE, Canvas, bgra_to_qpixmap
from bac_count.ui.file_list import FileList
from bac_count.ui.param_panel import ParamPanel
from bac_count.ui.presets import load_preset, save_preset
from bac_count.ui.worker import DetectionService

DEBOUNCE_MS = 150


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("平板菌落计数")
        self.resize(1440, 900)

        self.params: dict = default_params()
        self.image_params: dict[str, dict] = {}  # 每张图独立的参数快照
        self.images: list[Path] = []
        self.idx: int = -1
        self.histories: dict[str, EditHistory] = {}
        self.bgr: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.scale: float = 1.0
        self.last_auto: list[Colony] = []
        self._seq = 0
        self._busy = False
        self._pending = False

        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.canvas.emptyClicked.connect(self._on_add_click)
        self.canvas.colonyClicked.connect(self._on_remove_click)
        self.canvas.modeHintChanged.connect(self._update_mode_hint)
        self.canvas.pathsDropped.connect(lambda ps: self.load_paths(ps, append=True))

        self.file_list = FileList(self)
        self.file_list.pathsDropped.connect(lambda ps: self.load_paths(ps, append=True))
        self.file_list.deleteRequested.connect(self._delete_paths)
        self.file_list.clearRequested.connect(self._clear_list)
        self.file_list.copyParamsToSelected.connect(self._copy_params_to)
        self.file_list.copyParamsToAll.connect(self._copy_params_to_all)
        self.file_list.itemPicked.connect(self._on_list_pick)
        self._file_dock = QDockWidget("文件列表", self)
        self._file_dock.setObjectName("file_dock")
        self._file_dock.setWidget(self.file_list)
        self._file_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._file_dock)
        self._file_dock.setMinimumWidth(200)
        self.resizeDocks([self._file_dock], [230], Qt.Horizontal)

        self.panel = ParamPanel(self)
        self.panel.valueChanged.connect(self._on_param_changed)
        dock = QDockWidget("检测参数", self)
        dock.setObjectName("param_dock")
        dock.setWidget(self.panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(340)
        self.resizeDocks([dock], [380], Qt.Horizontal)

        self._build_toolbar()
        self._build_status()
        self._rebuild_panel()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.request_detection)

        self.service = DetectionService(self)
        self.service.worker.done.connect(self._on_detection_done)

    # ------------------------------------------------------------------ 工具栏
    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, checkable=False, group=None, button=True, tip=None) -> QAction:
            a = QAction(text, self)
            if callable(slot):
                a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            a.setCheckable(checkable)
            if group is not None:
                group.addAction(a)
            if button:
                tb.addAction(a)
            else:
                self.addAction(a)  # 不占工具栏，仅保留快捷键
            return a

        def menu_btn(text, items):
            a = QAction(text, self)
            m = QMenu(text, self)
            for label, slot in items:
                m.addAction(label, slot)
            a.setMenu(m)
            tb.addAction(a)
            btn = tb.widgetForAction(a)
            if btn is not None:
                btn.setPopupMode(QToolButton.InstantPopup)
            return a

        menu_btn("📂 打开", [("打开图片…", self.open_files), ("打开文件夹…", self.open_folder)])
        tb.addSeparator()
        act("◀", lambda: self._navigate(-1), "Alt+Left", tip="上一张 (Alt+←)")
        act("▶", lambda: self._navigate(1), "Alt+Right", tip="下一张 (Alt+→)")
        tb.addSeparator()
        self._a_filelist = act("🗂", self._toggle_file_list, "Ctrl+E", True, tip="文件列表 (Ctrl+E)")
        self._a_filelist.setChecked(True)
        self._file_dock.visibilityChanged.connect(lambda v: self._a_filelist.setChecked(v))
        tb.addSeparator()

        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self._a_browse = act("浏览", lambda: self._set_mode(MODE_BROWSE), "1", True, mode_group, tip="浏览模式 (1)")
        self._a_add = act("添加", lambda: self._set_mode(MODE_ADD), "2", True, mode_group, tip="添加菌落 (2)｜按住 Ctrl 临时删除")
        self._a_remove = act("删除", lambda: self._set_mode(MODE_REMOVE), "3", True, mode_group, tip="删除菌落 (3)｜按住 Ctrl 临时添加")
        self._a_browse.setChecked(True)
        tb.addSeparator()

        self._a_undo = act("↶", self._undo, "Ctrl+Z", tip="撤销 (Ctrl+Z)")
        self._a_redo = act("↷", self._redo, "Ctrl+Y", tip="重做 (Ctrl+Y)")
        self._a_undo.setEnabled(False)
        self._a_redo.setEnabled(False)
        tb.addSeparator()

        menu_btn("📤 导出", [("标注图 PNG", self.export_png), ("菌落明细 CSV", self.export_csv),
                             ("完整状态 JSON", self.export_json)])
        menu_btn("⚙ 预设", [("保存预设", self.save_preset), ("加载预设", self.load_preset),
                             ("恢复默认参数", self.reset_params), ("清除当前图编辑", self._clear_edits)])

        # 缩放快捷键保留（滚轮/0 已覆盖，不再占按钮）
        act("放大", lambda: self.canvas.zoom(1.25), QKeySequence.ZoomIn, button=False)
        act("缩小", lambda: self.canvas.zoom(1 / 1.25), QKeySequence.ZoomOut, button=False)
        act("适应窗口", self.canvas.fit_view, "0", button=False)

        # 工具栏按钮不抢焦点，保证空格/Ctrl 直达画布
        for btn in tb.findChildren(QToolButton):
            btn.setFocusPolicy(Qt.NoFocus)

    def _build_status(self) -> None:
        self._lbl_count = QLabel("计数: —")
        self._lbl_mode = QLabel("")
        self._lbl_detail = QLabel("")
        self._lbl_image = QLabel("未加载图片")
        sb = self.statusBar()
        sb.addWidget(self._lbl_count)
        sb.addWidget(self._lbl_mode, 1)
        sb.addWidget(self._lbl_detail)
        sb.addPermanentWidget(self._lbl_image)

    def _update_mode_hint(self, hint: str) -> None:
        self._lbl_mode.setText(hint)

    # ------------------------------------------------------------------ 打开
    def open_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMG_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "打开图片", str(Path.cwd()), f"图片 ({exts})")
        if files:
            self.load_paths([Path(f) for f in files])

    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "打开文件夹", str(Path.cwd()))
        if not d:
            return
        files = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)
        if not files:
            QMessageBox.information(self, "提示", "该文件夹内没有支持的图片。")
            return
        self.load_paths(files)

    def load_paths(self, paths: list[Path], append: bool = False) -> None:
        paths = [Path(p) for p in paths]
        if append:
            new = [p for p in paths if p not in self.images]
            if not new:
                return
            self.images += new
            if self.idx < 0 or self.bgr is None:
                self.idx = len(self.images) - len(new)  # 当前无图 → 直接看新导入的第一张
                self.file_list.set_paths(self.images)
                self.load_current()
            else:
                self._sync_list()
            return
        self.images = paths
        self.idx = 0
        self.file_list.set_paths(self.images)
        self.load_current()

    def _toggle_file_list(self) -> None:
        self._file_dock.setVisible(not self._file_dock.isVisible())

    def _sync_list(self) -> None:
        """结构变化（增删图片）时重建列表；纯导航不走这里。"""
        self.file_list.set_paths(self.images)
        self.file_list.mark_current(self.idx)

    def _on_list_pick(self, row: int) -> None:
        if 0 <= row < len(self.images) and row != self.idx:
            self.idx = row
            self.load_current()

    def _delete_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        old_current = self.images[self.idx] if 0 <= self.idx < len(self.images) else None
        self.images = [p for p in self.images if p not in set(paths)]
        if not self.images:
            self._reset_view()
            return
        self.idx = min(self.idx, len(self.images) - 1)
        self.file_list.set_paths(self.images)  # 结构变化先重建，再决定导航/标记
        if old_current in paths:
            self.load_current()
        else:
            self.file_list.mark_current(self.idx)

    def _clear_list(self) -> None:
        self._reset_view()

    def _reset_view(self) -> None:
        self.images = []
        self.idx = -1
        self.bgr = None
        self.alpha = None
        self.last_auto = []
        self.canvas.set_colonies([], 1.0)
        self.canvas.clear_image()
        self._lbl_count.setText("计数: —")
        self._lbl_detail.setText("")
        self._lbl_image.setText("未加载图片")
        self._sync_list()

    def _navigate(self, step: int) -> None:
        if not self.images:
            return
        self.idx = max(0, min(len(self.images) - 1, self.idx + step))
        self.load_current()

    def load_current(self) -> None:
        path = self.images[self.idx]
        try:
            self.bgr, self.alpha = load_image(path)
        except ValueError as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return
        # 每图独立参数：已存过程式 → 恢复；首次访问 → 继承上一张（即当前 params）
        key = str(path)
        if key in self.image_params:
            self.params = dict(self.image_params[key])
            self._rebuild_panel()
        else:
            self.image_params[key] = dict(self.params)
        h, w = self.bgr.shape[:2]
        self.scale = min(1.0, WORKING_MAX / max(h, w))
        if self.scale < 1.0:
            wb = cv2.resize(self.bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            wa = cv2.resize(self.alpha, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            wb, wa = self.bgr, self.alpha
        self.canvas.set_image(bgra_to_qpixmap(wb, wa))
        self.last_auto = []
        self._lbl_image.setText(f"{self.idx + 1}/{len(self.images)}  {path.name}  ({w}×{h})")
        self.file_list.mark_current(self.idx)  # 只移动 current，不重建、不销毁选区
        self._update_edit_actions()
        self.request_detection()

    # ------------------------------------------------------------------ 检测调度
    def request_detection(self) -> None:
        self._debounce.stop()
        if self.bgr is None:
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        self._seq += 1
        self.service.worker.submit.emit(self._seq, self.bgr, self.alpha, dict(self.params))

    def _on_detection_done(self, seq: int, colonies: list) -> None:
        if seq < self._seq:
            return  # 过期结果
        self.last_auto = colonies
        self._busy = False
        self._refresh_merge()
        if self._pending:
            self._pending = False
            self.request_detection()

    # ------------------------------------------------------------------ 编辑
    def _history(self) -> EditHistory:
        if not self.images or not (0 <= self.idx < len(self.images)):
            return EditHistory()
        return self.histories.setdefault(str(self.images[self.idx]), EditHistory())

    def _refresh_merge(self) -> None:
        if not self.images:
            return
        result = merge(self.last_auto, self._history().state)
        self.canvas.set_colonies(result.colonies, self.scale)
        self._lbl_count.setText(f"计数: {result.count}")
        self._lbl_detail.setText(
            f"自动 {result.auto_total}（保留 {result.kept} / 排除 {result.removed}）  手动 +{result.added}"
        )
        self._update_edit_actions()

    def _on_add_click(self, sx: float, sy: float) -> None:
        x, y = sx / self.scale, sy / self.scale
        self._history().add_colony(x, y, float(self.params.get("manual_r", 6.0)))
        self._refresh_merge()

    def _on_remove_click(self, colony: Colony) -> None:
        result = merge(self.last_auto, self._history().state)
        if self._history().remove_at(colony.x, colony.y, result.colonies, default_r=max(colony.r, 10)):
            self._refresh_merge()

    def _undo(self) -> None:
        if self._history().undo():
            self._refresh_merge()

    def _redo(self) -> None:
        if self._history().redo():
            self._refresh_merge()

    def _clear_edits(self) -> None:
        self._history().clear_edits()
        self._refresh_merge()

    def _update_edit_actions(self) -> None:
        h = self._history() if self.images else EditHistory()
        self._a_undo.setEnabled(h.can_undo())
        self._a_redo.setEnabled(h.can_redo())

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)

    # ------------------------------------------------------------------ 参数面板
    def _full_schema(self):
        from dataclasses import replace

        displays = [DETECTORS[k].display for k in DETECTORS]
        specs = [replace(s, choices=displays) if s.key == "algorithm" else s for s in PRE_SCHEMA]
        algo = self.params.get("algorithm", "adaptive")
        return specs + POST_SCHEMA + get_detector(algo).schema()

    def _rebuild_panel(self) -> None:
        values = dict(self.params)
        algo = values.get("algorithm", "adaptive")
        if algo in DETECTORS:
            values["algorithm"] = DETECTORS[algo].display
        self.panel.set_schema(self._full_schema(), values)

    def _on_param_changed(self, key: str, value) -> None:
        if key == "algorithm":
            display = value
            name = next((k for k in DETECTORS if DETECTORS[k].display == display), None)
            if name:
                self.params["algorithm"] = name
            self._commit_params()
            self._rebuild_panel()
            self.request_detection()
            return
        self.params[key] = value
        self._commit_params()
        if key == "manual_r":
            return  # 只影响手动添加，无需重检测
        self._debounce.start()

    def _commit_params(self) -> None:
        """把当前参数写回正在编辑的图片的快照。"""
        if self.images and 0 <= self.idx < len(self.images):
            self.image_params[str(self.images[self.idx])] = dict(self.params)

    def _copy_params_to(self, paths: list[Path]) -> None:
        if not (self.images and 0 <= self.idx < len(self.images)):
            return
        cur = str(self.images[self.idx])
        targets = [str(p) for p in paths if str(p) != cur]
        if not targets:
            self.statusBar().showMessage("⚠ 选中项里没有其他图片（只有当前图），未复制", 5000)
            return
        snap = dict(self.params)
        for t in targets:
            self.image_params[t] = dict(snap)
        self.statusBar().showMessage(
            f"已将「{self.images[self.idx].name}」的参数复制到 {len(targets)} 张图片", 5000)

    def _copy_params_to_all(self) -> None:
        if not self.images:
            return
        self._copy_params_to([p for i, p in enumerate(self.images) if i != self.idx])

    def save_preset(self) -> None:
        name, ok = _text_dialog(self, "保存预设", "预设名称：", "my_preset")
        if ok and name:
            p = save_preset(name, self.params)
            self.statusBar().showMessage(f"预设已保存: {p}", 4000)

    def load_preset(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMG_EXTS))
        f, _ = QFileDialog.getOpenFileName(self, "加载预设", str(Path.home() / ".bac_count" / "presets"), "JSON (*.json)")
        if not f:
            return
        try:
            data = load_preset(f)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))
            return
        self.params.update(data)
        self._commit_params()
        self._rebuild_panel()
        self.request_detection()

    def reset_params(self) -> None:
        self.params = default_params(self.params.get("algorithm", "adaptive"))
        self._commit_params()
        self._rebuild_panel()
        self.request_detection()

    # ------------------------------------------------------------------ 导出
    def _export_dir(self) -> Path:
        return self.images[self.idx].parent if self.images else Path.cwd()

    def _current_result(self):
        return merge(self.last_auto, self._history().state)

    def export_png(self) -> None:
        if self.bgr is None:
            return
        r = self._current_result()
        default = str(self._export_dir() / f"{self.images[self.idx].stem}_count{r.count}.png")
        f, _ = QFileDialog.getSaveFileName(self, "导出标注图", default, "PNG (*.png)")
        if f:
            export_png(f, self.bgr, r.colonies)
            self.statusBar().showMessage(f"已导出 {f}", 4000)

    def export_csv(self) -> None:
        if self.bgr is None:
            return
        r = self._current_result()
        default = str(self._export_dir() / f"{self.images[self.idx].stem}.csv")
        f, _ = QFileDialog.getSaveFileName(self, "导出 CSV", default, "CSV (*.csv)")
        if f:
            export_csv(f, r.colonies, self.images[self.idx].name)
            self.statusBar().showMessage(f"已导出 {f}", 4000)

    def export_json(self) -> None:
        if self.bgr is None:
            return
        r = self._current_result()
        default = str(self._export_dir() / f"{self.images[self.idx].stem}.json")
        f, _ = QFileDialog.getSaveFileName(self, "导出 JSON", default, "JSON (*.json)")
        if f:
            export_json(f, r.colonies, self._history().state, self.params, self.images[self.idx].name)
            self.statusBar().showMessage(f"已导出 {f}", 4000)

    def closeEvent(self, ev) -> None:
        self.service.stop()
        super().closeEvent(ev)


def _text_dialog(parent, title, label, default=""):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QVBoxLayout
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    lay = QVBoxLayout(dlg)
    le = QLineEdit(default)
    lay.addWidget(QLabel(label))
    lay.addWidget(le)
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    ok = dlg.exec() == QDialog.Accepted
    return le.text(), ok


def run() -> int:
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("bac_count")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            files = sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS)
            w.load_paths(files)
        elif p.exists():
            w.load_paths([p])
    return app.exec()
