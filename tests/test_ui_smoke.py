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
        w.file_list.itemPicked.emit(1)
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


def test_multiselect_survives_navigation(qtbot):
    """回归：Ctrl 加选 / Shift 扩选 / 右键选中项 都不应销毁选区或意外切图。"""
    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert len(files) >= 3
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    vp = fl.viewport()

    def pos(r):
        return fl.visualItemRect(fl.item(r)).center()

    def sel_rows():
        return {fl.row(i) for i in fl.selectedItems()}

    # 无修饰单击 row1 → 切图
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        qtbot.mouseClick(vp, Qt.LeftButton, pos=pos(1))
    qtbot.wait(200)
    assert w.idx == 1
    # Ctrl+单击 row2 → 加选，不切图
    qtbot.mouseClick(vp, Qt.LeftButton, Qt.ControlModifier, pos(2))
    qtbot.wait(100)
    assert w.idx == 1, "Ctrl 加选不应切换图片"
    assert sel_rows() == {1, 2}
    # Shift+单击 row0 → 范围选 {0,1,2}，不切图
    qtbot.mouseClick(vp, Qt.LeftButton, Qt.ShiftModifier, pos(0))
    qtbot.wait(100)
    assert w.idx == 1, "Shift 扩选不应切换图片"
    assert sel_rows() == {0, 1, 2}
    # 右键按在已选中项上 → 选区保留（只 press 不 release，避免弹菜单阻塞）
    qtbot.mousePress(vp, Qt.RightButton, pos=pos(2))
    qtbot.wait(100)
    assert sel_rows() == {0, 1, 2}, "右键不应破坏多选"
    assert w.idx == 1, "右键不应切换图片"
    w.close()


def _send_context_menu(widget, viewport_pos) -> None:
    """QTest 合成鼠标事件不会生成 QContextMenuEvent，手动发送。"""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication

    gp = widget.mapToGlobal(viewport_pos)
    ev = QContextMenuEvent(QContextMenuEvent.Reason.Mouse,
                           QPoint(viewport_pos), QPoint(gp))
    # QAbstractScrollArea 的 contextMenuEvent 由 viewport 事件转发触发
    QApplication.sendEvent(widget.viewport(), ev)


def test_rightclick_extends_selection_and_copies(qtbot, monkeypatch):
    """回归：用户报的 bug——右键未选中项会清空多选，导致"复制→选中项"目标为空。"""
    from bac_count.ui.file_list import FileList

    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert len(files) >= 4
    captured = []
    monkeypatch.setattr(FileList, "_popup_menu", lambda self, menu, pos: captured.append(menu))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    vp = fl.viewport()

    def pos(r):
        return fl.visualItemRect(fl.item(r)).center()

    def sel_rows():
        return {fl.row(i) for i in fl.selectedItems()}

    # 单击 row2 → 当前图=2；调参 blur_k=15
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        qtbot.mouseClick(vp, Qt.LeftButton, pos=pos(2))
    qtbot.wait(200)
    w._on_param_changed("blur_k", 15)
    # Ctrl 多选 0、1
    qtbot.mouseClick(vp, Qt.LeftButton, Qt.ControlModifier, pos(0))
    qtbot.mouseClick(vp, Qt.LeftButton, Qt.ControlModifier, pos(1))
    assert sel_rows() == {2, 0, 1}
    # 右键未选中的 row3 → 加选而非清选，且弹出菜单
    qtbot.mouseClick(vp, Qt.RightButton, pos=pos(3))
    assert sel_rows() == {2, 0, 1, 3}, f"右键破坏了选区: {sel_rows()}"
    _send_context_menu(fl, pos(3))
    assert captured, "右键没有触发上下文菜单"
    menu = captured[-1]
    act = next(a for a in menu.actions() if a.text().startswith("复制当前图参数 → 选中项"))
    assert "(3)" in act.text() and act.isEnabled()
    act.trigger()
    for r in (0, 1, 3):
        assert w.image_params[str(files[r])]["blur_k"] == 15, files[r].name
    assert str(files[4]) not in w.image_params  # 未选中未复制
    w.close()


def test_copy_to_selected_disabled_when_only_current(qtbot, monkeypatch):
    from bac_count.ui.file_list import FileList

    files = sorted(SAMPLE_DIR.glob("*.png"))
    captured = []
    monkeypatch.setattr(FileList, "_popup_menu", lambda self, menu, pos: captured.append(menu))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    vp = fl.viewport()
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        qtbot.mouseClick(vp, Qt.LeftButton, pos=fl.visualItemRect(fl.item(1)).center())
    qtbot.wait(200)
    # 选区只有当前图 → 目标数 0 → 菜单项置灰（不再是静默失败）
    qtbot.mouseClick(vp, Qt.RightButton, pos=fl.visualItemRect(fl.item(1)).center())
    _send_context_menu(fl, fl.visualItemRect(fl.item(1)).center())
    menu = captured[-1]
    act = next(a for a in menu.actions() if a.text().startswith("复制当前图参数 → 选中项"))
    assert "(0)" in act.text() and not act.isEnabled()
    w.close()


def test_export_batch(qtbot, monkeypatch, tmp_path):
    """批量导出：PNG→文件夹、CSV 一图一行、JSON 含路径/参数/菌落。"""
    import csv as _csv
    import json as _json

    from PySide6.QtWidgets import QFileDialog

    files = sorted(SAMPLE_DIR.glob("*.png"))[:2]
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(600)
    w._on_add_click(50, 50)  # 当前图加一个手动菌落
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "out"), "")))
    w.export_png()
    w.export_csv()
    w.export_json()
    assert len(list((tmp_path / "annotated").glob("*.png"))) == 2
    rows = list(_csv.reader(open(tmp_path / "out.csv", encoding="utf-8-sig")))
    assert rows[0][:2] == ["文件名", "菌落数量"] and len(rows) == 3  # 表头+2图
    data = _json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data["image_count"] == 2 and data["total_count"] == sum(i["count"] for i in data["images"])
    img0 = next(i for i in data["images"] if i["name"] == files[0].name)
    assert img0["path"] == str(files[0])
    assert {"params", "colonies", "edits"} <= set(img0)
    assert img0["count"] == len(img0["colonies"])
    assert any(c["source"] == "manual" for c in img0["colonies"])
    assert "algorithm" in img0["params"]
    w.close()


def test_view_modes(qtbot):
    from bac_count.ui.canvas import VIEW_MARK, VIEW_ORIG, VIEW_PREP

    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(600)
    c = w.canvas
    assert not c._prep_pix.pixmap().isNull(), "检测后应有预处理图"
    assert c._view_mode == VIEW_MARK
    assert c._pix.isVisible() and not c._prep_pix.isVisible()
    assert c._overlay and all(it.isVisible() for it in c._overlay)
    c.set_view_mode(VIEW_ORIG)
    assert c._pix.isVisible() and not c._prep_pix.isVisible()
    assert all(not it.isVisible() for it in c._overlay)
    c.set_view_mode(VIEW_PREP)
    assert c._prep_pix.isVisible() and not c._pix.isVisible()
    c.set_view_mode(VIEW_MARK)
    assert c._pix.isVisible() and not c._prep_pix.isVisible()
    assert c._view_btns[VIEW_MARK].isChecked()
    w.close()


def test_per_image_params(qtbot):
    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert len(files) >= 2
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    # 图0 调参 → 写入图0快照
    w._on_param_changed("min_area", 99)
    assert w.image_params[str(files[0])]["min_area"] == 99
    # 切到未访问的图1 → 继承当前参数
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.file_list.itemPicked.emit(1)
    qtbot.wait(300)
    assert w.params["min_area"] == 99
    # 图1 改成别的值
    w._on_param_changed("min_area", 7)
    # 切回图0 → 恢复图0自己的参数，面板同步
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.file_list.itemPicked.emit(0)
    qtbot.wait(300)
    assert w.params["min_area"] == 99
    assert w.panel.get_values()["min_area"] == 99
    w.close()


def test_copy_params(qtbot):
    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert len(files) >= 4
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    w._on_param_changed("min_area", 42)
    # 复制到指定多张
    w._copy_params_to(files[1:3])
    assert w.image_params[str(files[1])]["min_area"] == 42
    assert w.image_params[str(files[2])]["min_area"] == 42
    assert str(files[3]) not in w.image_params  # 未访问未复制 → 不受影响
    # 导航到目标图 → 用复制来的参数
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.file_list.itemPicked.emit(1)
    qtbot.wait(300)
    assert w.params["min_area"] == 42
    # 复制到全部其他
    w._on_param_changed("min_area", 11)
    w._copy_params_to_all()
    for i, p in enumerate(files):
        if i != 1:
            assert w.image_params[str(p)]["min_area"] == 11, p.name
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
