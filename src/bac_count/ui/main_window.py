"""主窗口：工具栏 + 左预览画布 + 右参数面板 + 状态栏。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QToolBar,
    QToolButton,
)

from bac_count.core.edits import EditHistory
from bac_count.core.export import colony_dict, edits_dict, export_png, write_full_json, write_summary_csv
from bac_count.core.models import Colony, EditState, merge
from bac_count.core.pipeline import POST_SCHEMA, PRE_SCHEMA, WORKING_MAX, default_params, run_detection
from bac_count.core.preprocess import IMG_EXTS, load_image
from bac_count.core.registry import DETECTORS, get_detector
from bac_count.ui.canvas import MODE_ADD, MODE_BROWSE, MODE_REMOVE, Canvas, bgra_to_qpixmap
from bac_count.ui.file_list import FileList
from bac_count.ui.param_panel import ParamPanel
from bac_count.ui.presets import load_preset, save_preset
from bac_count.ui.worker import DetectionService

DEBOUNCE_MS = 150

ICON_PATH = Path(__file__).resolve().parents[1] / "resources" / "logo.png"


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle("平板菌落计数")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
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
        self.canvas.pickPicked.connect(self._on_pick)
        self.panel_pick_key: str | None = None

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

        self.panel = ParamPanel(self, settings=settings)
        self.panel.valueChanged.connect(self._on_param_changed)
        self.panel.pickRequested.connect(self._start_pick)
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

        menu_btn("📤 导出", [("标注图 PNG（全部图→文件夹）", self.export_png),
                             ("汇总表 CSV（一图一行）", self.export_csv),
                             ("完整数据 JSON（全部图）", self.export_json)])
        menu_btn("⚙ 预设", [("保存预设", self.save_preset), ("加载预设", self.load_preset),
                             ("恢复默认参数", self.reset_params), ("清除当前图编辑", self._clear_edits)])

        # 缩放快捷键保留（滚轮/0 已覆盖，不再占按钮）
        act("放大", lambda: self.canvas.zoom(1.25), QKeySequence.ZoomIn, button=False)
        act("缩小", lambda: self.canvas.zoom(1 / 1.25), QKeySequence.ZoomOut, button=False)
        act("适应窗口", self.canvas.fit_view, "0", button=False)
        act("取消取色", self.canvas.cancel_pick, "Escape", button=False)

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
        # 切图即作废旧图的在途结果：_seq 必须在任何 await 之前推进，否则上一张图
        # 的检测完成时 seq 仍然相等，旧图的菌落会被画到新图画布上（几百毫秒内
        # 用户看到并可能截图记录的计数是错的）。
        self._seq += 1
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
            # JPG/BMP 等无 alpha 通道 → self.alpha 为 None（只在 >WORKING_MAX 的图上才会走到这里）
            wa = (cv2.resize(self.alpha, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
                  if self.alpha is not None else None)
        else:
            wb, wa = self.bgr, self.alpha
        self.canvas.set_image(bgra_to_qpixmap(wb, wa))
        self.canvas.set_prep_image(None)  # 切图先清预处理图，避免短暂显示上一张
        self.last_auto = []
        self._lbl_image.setText(f"{self.idx + 1}/{len(self.images)}  {path.name}  ({w}×{h})")
        self.file_list.mark_current(self.idx)  # 只移动 current，不重建、不销毁选区
        self._update_edit_actions()
        self.request_detection()

    # ------------------------------------------------------------------ 检测调度
    def _current_key(self) -> str | None:
        """当前图在 image_params/histories 里的键（无图时 None）。"""
        if self.images and 0 <= self.idx < len(self.images):
            return str(self.images[self.idx])
        return None

    def request_detection(self) -> None:
        self._debounce.stop()
        if self.bgr is None:
            return
        key = self._current_key()
        if key is None:
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        self._seq += 1
        self.service.worker.submit.emit(self._seq, key, self.bgr, self.alpha, dict(self.params))

    def _on_detection_done(self, seq: int, key: str, colonies: list, prep=None) -> None:
        # 线程空出来了——**不管结果过没过期都要复位**，否则过期结果回来时 _pending
        # 永远补不出去，界面会卡在"没有检测结果"上。
        self._busy = False
        # _seq 由"提交任务"和"切图"共同推进，代表"当前应该显示什么"的版本；
        # 再比对结果自带的图片键，双保险：旧图的标注不会画到新图上（串图）。
        if seq == self._seq and key == self._current_key():
            self.last_auto = colonies
            if prep is not None:
                self.canvas.set_prep_image(bgra_to_qpixmap(prep, None))
            self._refresh_merge()
        if self._pending:
            self._pending = False
            # 补发时重新读 self.params/当前图（故意不用提交时的快照）：
            # 用户在等结果期间又拖了滑杆，要的正是最新参数的结果。
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
        if self._history().remove_at(colony.x, colony.y, result.colonies):
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

    # ------------------------------------------------------------------ 取色
    def _start_pick(self, key: str) -> None:
        if self.bgr is None:
            self.statusBar().showMessage("⚠ 取色需要先打开图片", 4000)
            return
        self.panel_pick_key = key
        self.canvas.enter_pick()

    def _on_pick(self, r: int, g: int, b: int) -> None:
        key = self.panel_pick_key
        if not key:
            return
        cur = list(self.params.get(key) or [])
        cur.append([int(r), int(g), int(b)])
        self.params[key] = cur
        self.panel.set_values({key: cur})
        self._commit_params()
        if key in ("keep_colors", "drop_colors") and not self.params.get("color_on"):
            # 颜色过滤取色后自动开总开关；HSV 取色无需开关（空表=不检测）
            self.params["color_on"] = True
            self.panel.set_values({"color_on": True})
            self.statusBar().showMessage("已自动开启颜色过滤", 4000)
        self._debounce.start()

    # ------------------------------------------------------------------ 参数面板
    def _full_schema(self):
        from dataclasses import replace

        displays = [DETECTORS[k].display for k in DETECTORS]
        specs = [replace(s, choices=displays) if s.key == "algorithm" else s for s in PRE_SCHEMA]
        algo = self.params.get("algorithm", "adaptive")
        # 面板顺序 = 执行顺序：皿ROI/预处理 → 检测算法(选择+参数) → 拆分 → 颜色过滤 → 过滤 → 手动
        return specs + get_detector(algo).schema() + POST_SCHEMA

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
                # 补齐该算法的默认参数——否则检测线程 KeyError（hsv 崩溃根因）
                for s in get_detector(name).schema():
                    if s.key not in self.params:
                        self.params[s.key] = list(s.default) if isinstance(s.default, list) else s.default
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

    # ------------------------------------------------------------------ 导出（全列表批量）
    def _export_dir(self) -> Path:
        return self.images[self.idx].parent if self.images else Path.cwd()

    def _current_result(self):
        return merge(self.last_auto, self._history().state)

    def _params_for_path(self, p: Path) -> tuple[dict, bool]:
        """该图该用哪套参数，并回报它是不是"从未打开过"的兜底。

        有快照 → 快照；从未打开过 → **沿用当前图参数**。README 里"新图继承上一张
        的参数"就是这个语义，批量导出必须与用户逐张点开时看到的数一致
        （旧写法兜底 `default_params()` → 导出数≠所见数，静默出错）。
        """
        snap = self.image_params.get(str(p))
        if snap is not None:
            return dict(snap), False
        return dict(self.params), True

    def _result_for_path(self, p: Path, decoded: tuple | None = None):
        """任意列表图的最终结果：当前图用内存态；其他图按其参数即时重算。

        `decoded` 传入已解码的 (bgr, alpha) 时直接复用——否则批量导出标注图会
        对每张图解码两遍（4000px 手机照片一次 JPEG 解码实测 84.5ms）。
        返回 (DetectionResult, params_used)。
        """
        key = str(p)
        if self.images and 0 <= self.idx < len(self.images) and self.images[self.idx] == p:
            return self._current_result(), dict(self.params)
        params, _inherited = self._params_for_path(p)
        bgr, alpha = decoded if decoded is not None else load_image(p)
        auto, _ctx, _s = run_detection(bgr, alpha, params)
        hist = self.histories.get(key)
        result = merge(auto, hist.state if hist else EditState())
        return result, params

    def _untuned_images(self) -> list[Path]:
        """列表里从未被打开过（因此没有参数快照）的图。"""
        return [p for p in self.images if str(p) not in self.image_params]

    def _confirm_inherited(self, missing: list[Path]) -> bool:
        """导出前把"这些图将沿用当前图参数"这件事讲到明面上（数据可信度问题）。"""
        if not missing:
            return True
        names = "、".join(p.name for p in missing[:3])
        if len(missing) > 3:
            names += f" 等 {len(missing)} 张"
        ret = QMessageBox.question(
            self, "确认导出参数",
            f"有 {len(missing)} 张图片从未在窗口里打开过：{names}。\n\n"
            "这些图将沿用**当前图片的参数**重新检测（与逐张点开看到的结果一致），"
            "而不是出厂默认参数。\n\n是否继续导出？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return ret == QMessageBox.Yes

    def _finish_export(self, done: int, failures: list[str], target: str | Path) -> None:
        """批量导出收尾：失败必须让用户看见。

        只 `print` 的话——打包成 exe 后 `--windows-console-mode=disable`
        根本没有控制台（磁盘满/文件被占/编码不支持全成静默失败），
        所以这里一律走状态栏 + QMessageBox。
        """
        if failures:
            detail = "\n".join(failures[:8])
            if len(failures) > 8:
                detail += f"\n…（另有 {len(failures) - 8} 条）"
            QMessageBox.warning(self, f"{len(failures)} 项失败",
                                f"成功 {done} 张，失败 {len(failures)} 张：\n\n{detail}")
            self.statusBar().showMessage(f"导出到 {target}：成功 {done} 张，失败 {len(failures)} 张", 8000)
        else:
            self.statusBar().showMessage(f"已导出 {done} 张到 {target}", 6000)

    def _iter_with_progress(self, title: str):
        """带进度/取消的批量迭代器；取消则提前结束。"""
        dlg = QProgressDialog(title, "取消", 0, len(self.images), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(200)
        for i, p in enumerate(self.images):
            dlg.setValue(i)
            QApplication.processEvents()
            if dlg.wasCanceled():
                break
            yield p
        dlg.setValue(dlg.maximum())
        dlg.close()

    def export_png(self) -> None:
        if not self.images:
            return
        folder = QFileDialog.getExistingDirectory(self, "导出标注图到文件夹", str(self._export_dir()))
        if not folder:
            return
        if not self._confirm_inherited(self._untuned_images()):
            return
        out = Path(folder) / "annotated"
        try:
            out.mkdir(exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "导出失败", f"无法创建目录 {out}：{e}")
            return
        n = 0
        failures: list[str] = []
        for p in self._iter_with_progress("正在导出标注图…"):
            try:
                bgr, alpha = load_image(p)
                result, _ = self._result_for_path(p, decoded=(bgr, alpha))
                export_png(out / f"{p.stem}_count{result.count}.png", bgr, result.colonies)
                n += 1
            except Exception as e:
                failures.append(f"{p.name}: {e}")
        self._finish_export(n, failures, out)

    def export_csv(self) -> None:
        if not self.images:
            return
        f, _ = QFileDialog.getSaveFileName(self, "导出汇总 CSV", str(self._export_dir() / "counts_summary.csv"), "CSV (*.csv)")
        if not f:
            return
        if not self._confirm_inherited(self._untuned_images()):
            return
        rows = []
        failures: list[str] = []
        for p in self._iter_with_progress("正在汇总…"):
            try:
                result, _ = self._result_for_path(p)
                rows.append({"name": p.name, "count": result.count, "auto_total": result.auto_total,
                             "added": result.added, "removed": result.removed})
            except Exception as e:
                failures.append(f"{p.name}: {e}")
        try:
            write_summary_csv(f, rows)
        except Exception as e:  # 写盘失败也摆到明面上，不让整批工作静默作废
            QMessageBox.critical(self, "写入失败", f"{f}\n\n{e}")
            return
        self._finish_export(len(rows), failures, f)

    def export_json(self) -> None:
        if not self.images:
            return
        f, _ = QFileDialog.getSaveFileName(self, "导出完整 JSON", str(self._export_dir() / "bac_count_export.json"), "JSON (*.json)")
        if not f:
            return
        if not self._confirm_inherited(self._untuned_images()):
            return
        items = []
        failures: list[str] = []
        for p in self._iter_with_progress("正在导出完整数据…"):
            try:
                result, params = self._result_for_path(p)
                hist = self.histories.get(str(p))
                items.append({
                    "path": str(p),
                    "name": p.name,
                    "count": result.count,
                    "params": params,
                    "colonies": [colony_dict(c) for c in result.colonies],
                    "edits": edits_dict(hist.state if hist else EditState()),
                })
            except Exception as e:
                failures.append(f"{p.name}: {e}")
        try:
            write_full_json(f, items)
        except Exception as e:
            # 这里在批量循环**之外**：不收口的话"收集完所有图才炸"= 前面全部工作作废、
            # 文件一个字没写，而且异常落在 GUI 线程上（无控制台的 exe 里行为未定义）。
            QMessageBox.critical(self, "写入失败", f"{f}\n\n{e}")
            return
        self._finish_export(len(items), failures, f)

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
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))  # 任务栏/Dock 图标
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
