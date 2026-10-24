"""GUI 冒烟测试（offscreen）：构建窗口、加载样例、参数变更、手动增删。"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt

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


def test_space_temporary_pan(qtbot):
    from PySide6.QtWidgets import QGraphicsView

    w = MainWindow()
    qtbot.addWidget(w)
    c = w.canvas
    from bac_count.ui.canvas import MODE_ADD

    w._set_mode(MODE_ADD)
    assert c.dragMode() == QGraphicsView.NoDrag
    qtbot.keyPress(c, Qt.Key_Space)
    assert c._space_pan and c.dragMode() == QGraphicsView.ScrollHandDrag
    qtbot.keyRelease(c, Qt.Key_Space)
    assert not c._space_pan and c.dragMode() == QGraphicsView.NoDrag
    w.close()


def test_ctrl_toggles_add_remove(qtbot):
    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(500)
    c = w.canvas
    from bac_count.ui.canvas import MODE_ADD, MODE_REMOVE

    w._set_mode(MODE_ADD)
    assert c._effective_mode() == MODE_ADD
    qtbot.keyPress(c, Qt.Key_Control)
    assert c._effective_mode() == MODE_REMOVE
    # Ctrl+点击已有菌落 → 删除（而非添加）
    target = w._current_result().colonies[0]
    base = w._current_result().count
    pos = c.mapFromScene(QPointF(target.x * c._scale, target.y * c._scale))
    qtbot.mouseClick(c.viewport(), Qt.LeftButton, pos=pos)
    assert w._current_result().count == base - 1
    qtbot.keyRelease(c, Qt.Key_Control)
    assert c._effective_mode() == MODE_ADD
    # 松开 Ctrl 后回到添加模式：点击空白 → 添加
    empty = c.mapFromScene(QPointF(5, 5))
    qtbot.mouseClick(c.viewport(), Qt.LeftButton, pos=empty)
    assert w._current_result().count == base
    w.close()


def test_file_list_ops(qtbot):
    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert len(files) >= 2
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    assert w.file_list.count() == len(files)
    assert w.file_list.currentRow() == 0
    # 追加去重
    w.load_paths(files[:2], append=True)
    assert len(w.images) == len(files)
    # 列表点击切换图片
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.file_list.setCurrentRow(1)
    qtbot.wait(300)
    assert w.idx == 1
    # 删除当前图片 → 自动切到下一张
    cur = w.images[1]
    w._delete_paths([cur])
    qtbot.wait(300)
    assert cur not in w.images
    assert w.file_list.count() == len(files) - 1
    assert w.bgr is not None
    # 清空列表
    w._clear_list()
    assert not w.images and w.bgr is None and w.file_list.count() == 0
    # dock 折叠
    w._toggle_file_list()
    assert not w._file_dock.isVisible()
    w.close()


def test_extract_image_paths(tmp_path):
    from bac_count.ui.file_list import extract_image_paths
    from PySide6.QtCore import QUrl

    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpg").write_bytes(b"x")
    urls = [QUrl.fromLocalFile(str(tmp_path / "a.png")), QUrl.fromLocalFile(str(tmp_path / "b.txt")),
            QUrl.fromLocalFile(str(sub))]
    got = extract_image_paths(urls)
    assert [p.name for p in got] == ["a.png", "c.jpg"]
