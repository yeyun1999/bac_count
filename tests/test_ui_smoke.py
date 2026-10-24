"""GUI 冒烟测试（offscreen）：构建窗口、加载样例、参数变更、手动增删。"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from bac_count.ui.canvas import MODE_ADD, MODE_REMOVE
from bac_count.ui.main_window import MainWindow

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample"


def test_load_and_detect(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert files, "sample/ 无图片"
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(500)
    assert w.bgr is not None
    w.close()


def test_param_change_triggers_redetect(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    files = sorted(SAMPLE_DIR.glob("*.png"))
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w._on_param_changed("min_area", 50)
        w.request_detection()
    qtbot.wait(300)
    w.close()


def test_manual_add_remove_flow(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    files = sorted(SAMPLE_DIR.glob("*.png"))
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(600)  # 等首轮结果合并
    base = w._current_result().count
    # 添加
    w._on_add_click(50, 50)
    assert w._current_result().count == base + 1
    # 撤销
    w._undo()
    assert w._current_result().count == base
    # 重做
    w._redo()
    assert w._current_result().count == base + 1
    # 删除刚加的手动菌落
    added = [c for c in w._current_result().colonies if c.source == "manual"][0]
    w._on_remove_click(added)
    assert w._current_result().count == base
    w.close()


def test_algorithm_switch(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    files = sorted(SAMPLE_DIR.glob("*.png"))
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    for display in ("全局阈值+轮廓", "Blob 检测", "HSV 颜色分割", "Hough 圆"):
        with qtbot.waitSignal(w.service.worker.done, timeout=20000):
            w._on_param_changed("algorithm", display)
    assert w.params["algorithm"] == "hough"
    w.close()
