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


def test_param_tooltips_and_default_collapse(qtbot, tmp_path):
    from PySide6.QtCore import QSettings

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    # 每个参数都有悬浮解释，且已设到控件 tooltip
    for spec in w._full_schema():
        assert spec.desc, f"{spec.key} 缺少解释文案"
    _, blur_w = w.panel._widgets["blur_k"]
    assert blur_w.toolTip() == next(s for s in w._full_schema() if s.key == "blur_k").desc
    # 分组顺序 = 检测执行顺序
    assert list(w.panel._boxes) == [
        "皿 ROI", "预处理", "检测算法", "粘连拆分", "颜色过滤", "后处理过滤", "手动校正"]
    # 默认折叠策略：预处理/后处理过滤/手动校正收起，其余展开
    boxes = w.panel._boxes
    assert boxes["预处理"]._collapsed and boxes["后处理过滤"]._collapsed and boxes["手动校正"]._collapsed
    assert not boxes["皿 ROI"]._collapsed and not boxes["检测算法"]._collapsed and not boxes["粘连拆分"]._collapsed
    w.close()


def test_collapse_memory(qtbot, tmp_path):
    from PySide6.QtCore import QSettings

    from bac_count.core.models import ParamSpec
    from bac_count.ui.param_panel import ParamPanel

    settings = QSettings(str(tmp_path / "mem.conf"), QSettings.Format.IniFormat)
    specs = [ParamSpec("x1", "参数一", "int", 5, group="测试折叠组X", min=0, max=10, desc="d")]
    p = ParamPanel(settings=settings)
    p.set_schema(specs, {})
    box = p._boxes["测试折叠组X"]
    assert not box._collapsed  # 未知组默认展开
    box.header.click()        # 用户折叠
    assert box._collapsed and not box.content.isVisibleTo(p)
    assert str(settings.value("collapsed/测试折叠组X")).lower() == "true"
    p2 = ParamPanel(settings=settings)
    p2.set_schema(specs, {})
    assert p2._boxes["测试折叠组X"]._collapsed  # 新面板恢复折叠状态
    p2._boxes["测试折叠组X"].header.click()
    assert p2._boxes["测试折叠组X"].content.isVisibleTo(p2)


def test_export_batch(qtbot, monkeypatch, tmp_path):
    """批量导出：PNG→文件夹、CSV 一图一行、JSON 含路径/参数/菌落。"""
    import csv as _csv
    import json as _json
    from PySide6.QtWidgets import QFileDialog, QMessageBox

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
    # 未打开过的图会先弹一次"沿用当前图参数"确认框：记下文案再放行
    asks: list[str] = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: asks.append(a[2]) or QMessageBox.Yes))
    w.export_png()
    w.export_csv()
    w.export_json()
    assert len(asks) == 3 and all("从未在窗口里打开过" in t for t in asks)
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


def test_view_modes_and_marks_toggle(qtbot):
    from bac_count.ui.canvas import VIEW_ORIG, VIEW_PREP

    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(600)
    c = w.canvas
    assert not c._prep_pix.pixmap().isNull(), "检测后应有预处理图"
    assert c._view_mode == VIEW_ORIG and c._show_marks
    assert c._pix.isVisible() and not c._prep_pix.isVisible()
    assert c._overlay and all(it.isVisible() for it in c._overlay)
    # 原图视图关标记 → 只剩干净原图
    c._chk_marks.setChecked(False)
    assert all(not it.isVisible() for it in c._overlay)
    assert c._pix.isVisible()
    # 预处理视图 + 标记开：灰度产物与菌落圈同时可见
    c._chk_marks.setChecked(True)
    c.set_view_mode(VIEW_PREP)
    assert c._prep_pix.isVisible() and not c._pix.isVisible()
    assert c._overlay and all(it.isVisible() for it in c._overlay)
    # 预处理视图也能独立关标记
    c._chk_marks.setChecked(False)
    assert all(not it.isVisible() for it in c._overlay)
    assert c._prep_pix.isVisible()
    # 无预处理图时，预处理视图回退显示原图
    c._chk_marks.setChecked(True)
    c.set_prep_image(None)
    assert c._pix.isVisible() and not c._prep_pix.isVisible()
    c.set_view_mode(VIEW_ORIG)
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


def test_color_pick_flow(qtbot, tmp_path):
    """端到端：🎨 → 画布点击取色 → 过滤生效 → 色块删除 → 取消取色。"""
    import cv2
    import numpy as np
    from PySide6.QtCore import QSettings

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (210, 235, 240)  # BGR 浅黄底
    for x, y in [(80, 90), (150, 60), (220, 90)]:
        cv2.circle(img, (x, y), 10, (60, 60, 210), -1)   # 红菌落
    for x, y in [(100, 210), (200, 210)]:
        cv2.circle(img, (x, y), 10, (60, 200, 60), -1)   # 绿菌落
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(300)
    assert len(w.last_auto) == 5

    w.panel.pickRequested.emit("keep_colors")
    assert w.canvas._picking and w.panel_pick_key == "keep_colors"

    pos = w.canvas.mapFromScene(QPointF(80, 90))  # scale=1，场景=工作=原图像素
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        qtbot.mouseClick(w.canvas.viewport(), Qt.LeftButton, pos=pos)
    assert w.params["keep_colors"] == [[210, 60, 60]]
    assert w.params["color_on"] is True      # 取色后自动开启过滤
    assert not w.canvas._picking             # 单击取一个即退出
    qtbot.wait(400)
    assert len(w.last_auto) == 3, f"应只留红色，实得 {len(w.last_auto)}"

    _, widget = w.panel._widgets["keep_colors"]
    assert widget.get_value() == [[210, 60, 60]]
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        widget._remove([210, 60, 60])
    qtbot.wait(300)
    assert w.params["keep_colors"] == []
    assert len(w.last_auto) == 5

    w.panel.pickRequested.emit("drop_colors")
    assert w.canvas._picking
    qtbot.mouseClick(w.canvas.viewport(), Qt.RightButton,
                     pos=w.canvas.mapFromScene(QPointF(100, 210)))
    assert not w.canvas._picking and w.params["drop_colors"] == []
    w.close()


def test_pick_space_pan_and_loupe(qtbot, tmp_path):
    """取色中：空格临时平移不吞点击；放大镜跟随并显示正确颜色。"""
    import cv2
    import numpy as np
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QGraphicsView

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (210, 235, 240)
    cv2.circle(img, (80, 90), 10, (60, 60, 210), -1)
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(200)
    c = w.canvas
    w.panel.pickRequested.emit("keep_colors")
    assert c._picking

    pos = c.mapFromScene(QPointF(80, 90))
    qtbot.mouseMove(c.viewport(), pos=pos)
    assert c._loupe.isVisibleTo(c.viewport())
    assert c._loupe._rgb == (210, 60, 60)
    assert c.dragMode() == QGraphicsView.NoDrag

    qtbot.keyPress(c, Qt.Key_Space)
    assert c.dragMode() == QGraphicsView.ScrollHandDrag
    assert not c._loupe.isVisible()
    n0 = len(w.params["keep_colors"])
    qtbot.mouseClick(c.viewport(), Qt.LeftButton, pos=pos)
    assert len(w.params["keep_colors"]) == n0, "空格平移期间的点击不应取色"

    qtbot.keyRelease(c, Qt.Key_Space)
    assert c._picking and c.dragMode() == QGraphicsView.NoDrag

    with qtbot.waitSignal(c.pickPicked, timeout=5000):
        qtbot.mouseClick(c.viewport(), Qt.LeftButton, pos=pos)
    assert not c._picking and not c._loupe.isVisible()
    assert w.params["keep_colors"] == [[210, 60, 60]]
    w.close()


def test_pick_focus_from_panel_button(qtbot, tmp_path):
    """真实链路回归：点面板 🎨 后画布必须拿到键盘焦点，空格立刻可临时平移。"""
    import cv2
    import numpy as np
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QGraphicsView

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (210, 235, 240)
    cv2.circle(img, (80, 90), 10, (60, 60, 210), -1)
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(200)
    _, widget = w.panel._widgets["keep_colors"]
    qtbot.mouseClick(widget._btn_pick, Qt.LeftButton)  # 真实点击：焦点原本会被按钮抢走
    assert w.canvas._picking
    assert w.canvas.hasFocus(), "进入取色后画布应拿到键盘焦点（空格直达画布）"
    qtbot.keyPress(w.canvas, Qt.Key_Space)
    assert w.canvas.dragMode() == QGraphicsView.ScrollHandDrag
    n0 = len(w.params["keep_colors"])
    qtbot.mouseClick(w.canvas.viewport(), Qt.LeftButton,
                     pos=w.canvas.mapFromScene(QPointF(80, 90)))
    assert len(w.params["keep_colors"]) == n0, "空格平移期间不应取色"
    w.close()


def test_loupe_tracks_view_zoom(qtbot, tmp_path):
    """放大镜源区域随视图缩放自适应（放大视图后放大镜跟着变近）。"""
    import cv2
    import numpy as np
    from PySide6.QtCore import QSettings

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (210, 235, 240)
    cv2.circle(img, (80, 90), 10, (60, 60, 210), -1)
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(200)
    c = w.canvas
    c.enter_pick()
    c._update_loupe(QPointF(80, 90), QPointF(100, 100))
    s1 = c._loupe._src
    c.zoom(2.0)
    c._update_loupe(QPointF(80, 90), QPointF(100, 100))
    s2 = c._loupe._src
    assert s2 < s1 / 1.5, f"视图放大2x后源区域应约减半：{s1:.1f}→{s2:.1f}"
    for _ in range(8):
        c.zoom(0.2)
        c._update_loupe(QPointF(80, 90), QPointF(100, 100))
    assert c._loupe._src <= 160.0, "远端应被钳制"
    c.cancel_pick()
    w.close()


def test_pick_wheel_zoom_anchor(qtbot, tmp_path):
    """取色中滚轮缩放必须以光标为锚点（光标下场景点不动）——回归 move 未转发基类导致锚点漂移。"""
    import cv2
    import numpy as np
    from PySide6.QtCore import QPoint, QPointF, QSettings
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (210, 235, 240)
    cv2.circle(img, (80, 90), 10, (60, 60, 210), -1)
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(200)
    c = w.canvas
    c.enter_pick()
    vp = QPoint(180, 180)
    qtbot.mouseMove(c.viewport(), pos=vp)
    before = c.mapToScene(vp)
    ev = QWheelEvent(QPointF(vp), c.viewport().mapToGlobal(vp),
                     QPoint(0, 0), QPoint(0, 120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(c.viewport(), ev)
    after = c.mapToScene(vp)
    assert (after - before).manhattanLength() < 1.0, f"光标锚点漂移: {before} → {after}"
    c.cancel_pick()
    w.close()


def test_load_large_jpg_without_alpha(qtbot, tmp_path):
    """>WORKING_MAX 且无 alpha 通道的 JPG：缩放分支不得对 None alpha 调 cv2.resize。

    回归：main_window.load_current 里 cv2.resize(self.alpha, ...) 未判空，
    手机拍的照片（JPG、>1200px）一打开就抛 OpenCV (-215:Assertion failed) !ssize.empty()。
    """
    import cv2
    import numpy as np

    from bac_count.core.pipeline import WORKING_MAX

    p = tmp_path / "big.jpg"
    img = np.full((WORKING_MAX + 400, WORKING_MAX + 300, 3), 210, np.uint8)
    cv2.circle(img, (600, 700), 18, (120, 110, 100), -1)
    assert cv2.imwrite(str(p), img)

    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=30000):
        w.load_paths([p])
    assert w.alpha is None, "JPG 不该有 alpha"
    assert w.scale < 1.0, "图应被缩到工作尺寸（才会触发缩放分支）"
    assert w.bgr is not None
    w.close()


def test_stale_result_not_applied_across_image_switch(qtbot, monkeypatch):
    """检测在途时切图：旧图返回的结果必须被丢弃，不能当成新图的结果（串图）。

    回归：`load_current` 不推进 `_seq`、`request_detection` 忙时也不自增 →
    旧任务回来时 `seq < self._seq` 为假 → 图1的 51 个菌落被画到图2的画布上，
    状态栏也是图1的计数；用户在几百毫秒里看到（并可能截图记录）的数是错的。

    采样点很关键：同一信号有多个接收者时 Qt 不保证调用顺序（实测后连接的 lambda
    会先于 MainWindow 的槽跑），所以直接轮询 `w.last_auto` 会"看起来通过"而什么
    都没验证——这里把 `_on_detection_done` 包一层，在**槽执行完之后**采样。
    """
    import time

    from PySide6.QtWidgets import QApplication

    from bac_count.core.pipeline import default_params, run_detection
    from bac_count.core.preprocess import load_image

    files = sorted(SAMPLE_DIR.glob("*.png"))
    a, b = files[0], files[1]
    seen: list[tuple] = []  # (seq, key, 返回的菌落数, 槽执行完后的 last_auto 数)
    real_slot = MainWindow._on_detection_done

    def spy(self, seq, key, colonies, prep=None):
        real_slot(self, seq, key, colonies, prep)
        seen.append((seq, key, len(colonies), len(self.last_auto)))

    monkeypatch.setattr(MainWindow, "_on_detection_done", spy)  # 建窗口前替换：__init__ 连的就是 spy
    w = MainWindow()
    qtbot.addWidget(w)
    w.load_paths([a])  # 不 processEvents → a 的 done 还排在队列里，_busy 已置真
    assert w._busy, "a 的检测应仍在途"
    w.load_paths([b])  # 在途期间切图

    t = time.time()
    while not seen and time.time() - t < 30:
        QApplication.processEvents()
        time.sleep(0.005)
    assert seen, "worker 没有回任何结果"
    _seq_a, key_a, n_a, applied = seen[0]
    assert key_a == str(a) and n_a > 0, "先回来的应当是图 a 的非空结果"
    assert applied == 0, f"旧图结果被采纳了（last_auto={applied}）→ 图1的标注会画到图2上"

    expect = len(run_detection(*load_image(b), default_params(w.params["algorithm"]))[0])
    t = time.time()  # b 的补发结果最终落地（顺带证明 _busy/_pending 没被过期结果卡死）
    while time.time() - t < 30:
        QApplication.processEvents()
        if any(k == str(b) and n == expect for _s, k, _r, n in seen):
            break
        time.sleep(0.005)
    assert any(k == str(b) and n == expect for _s, k, _r, n in seen), f"b 的结果没落地: {seen}"
    assert w._lbl_count.text() == f"计数: {expect}"
    w.close()


def test_detection_guard_drops_stale_but_resets_busy(qtbot):
    """`_on_detection_done` 的两条丢弃条件 + 无条件复位 `_busy`（确定性，不靠线程时序）。

    ① seq 相等但图片键不是当前图 —— 修复前正是这一条被漏掉（切图不推进 _seq）；
    ② seq 已被更新的任务推进；
    ③ 丢弃时 `_busy` 照样要复位，否则 `_pending` 那条补发路被堵死，
       界面永远停在"没有检测结果"（把 `self._busy = False` 挪回 if 里面即复现）。
    """
    from bac_count.core.models import Colony

    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files[:1])
    qtbot.wait(600)
    assert w.last_auto and w._lbl_count.text().startswith("计数:")
    original = w.last_auto

    # 切到另一张图：seq 推进 + 当前键换成 files[1]
    w.images = files[:2]
    w.idx = 1
    w._seq += 1
    stale = [Colony(x=1.0, y=1.0, r=2.0) for _ in range(3)]

    w._busy, w._pending = True, False
    w._on_detection_done(w._seq, str(files[0]), stale, None)  # ① seq 相同、图不对
    assert w.last_auto is original, "旧图结果被采纳 → 串图"
    assert not w._busy, "过期结果回来后线程状态没复位"

    w._busy, w._pending = True, False
    w._on_detection_done(w._seq - 1, str(files[1]), stale, None)  # ② 更旧的 seq
    assert w.last_auto is original

    w._busy, w._pending = True, True
    w._on_detection_done(w._seq, str(files[1]), stale, None)  # ③ 采纳并补发
    assert w.last_auto is stale and w._lbl_count.text() == f"计数: {len(stale)}"
    assert w._busy and not w._pending, "补发没被触发"
    w.close()


def test_export_untuned_image_inherits_current_params(qtbot, monkeypatch, tmp_path):
    """没点开过的图：批量导出沿用**当前图参数**（=逐张点开会看到的数），而不是出厂默认。

    回归：_result_for_path 的 `or default_params()` 兜底，让"导入50张→只调了3张→导出"
    剩下 47 张用出厂参数重算，导出数≠所见数，且完全静默。
    """
    import json as _json

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    files = sorted(SAMPLE_DIR.glob("*.png"))[:3]
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(600)
    # 把当前图（files[0]）调成与出厂默认明显不同的一套
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w._on_param_changed("algorithm", "Hough 圆")
    w._on_param_changed("min_area", 500)
    w._commit_params()

    tuned, inherited0 = w._params_for_path(files[0])
    assert not inherited0 and tuned["algorithm"] == "hough" and tuned["min_area"] == 500
    for p in files[1:]:
        got, inherited = w._params_for_path(p)
        assert inherited, f"{p.name} 从未打开过，应标记为兜底"
        assert got == tuned, "未调参的图应沿用当前图参数（README：新图继承上一张的参数）"

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "out"), "")))
    asks: list[str] = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: asks.append(a[2]) or QMessageBox.Yes))
    w.export_json()
    assert len(asks) == 1 and "2 张" in asks[0], "应就 2 张未调参的图征求确认"
    data = _json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    for item in data["images"]:
        assert item["params"]["algorithm"] == "hough" and item["params"]["min_area"] == 500
    w.close()


def test_export_reports_per_image_failures(qtbot, monkeypatch, tmp_path):
    """批量导出里单张失败：弹窗 + 状态栏汇总，其余图照常导出。

    回归：三处 `except Exception: print(...)`——打包成 exe 后
    `--windows-console-mode=disable` 没有控制台，磁盘满/文件被占全是静默失败。
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    files = sorted(SAMPLE_DIR.glob("*.png"))[:2]
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(400)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    warns: list[tuple] = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warns.append(a) or QMessageBox.StandardButton.Ok))

    real = MainWindow._result_for_path

    def flaky(self, p, decoded=None):
        if Path(p).name == files[1].name:
            raise OSError("模拟：磁盘已满")
        return real(self, p, decoded=decoded)

    monkeypatch.setattr(MainWindow, "_result_for_path", flaky)
    w.export_png()  # 不许抛到 Qt 槽外
    assert len(warns) == 1 and "1 项失败" in warns[0][1], warns
    assert "磁盘已满" in warns[0][2]
    out = list((tmp_path / "annotated").glob("*.png"))
    assert len(out) == 1, f"另一张图应照常导出，实得 {out}"
    assert "失败 1" in w.statusBar().currentMessage()
    w.close()


def test_json_write_failure_is_caught(qtbot, monkeypatch, tmp_path):
    """`write_full_json` 在批量循环**之外**：异常必须被接住并告知用户。

    回归：它原来裸在外面（GUI 线程未捕获异常，PySide6 把 traceback 打到 stderr，
    无控制台的 exe 里行为未定义），而且前面所有图的收集工作全部作废。
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    import bac_count.ui.main_window as mw

    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(sorted(SAMPLE_DIR.glob("*.png"))[:1])
    qtbot.wait(400)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "out"), "")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    criticals: list[tuple] = []
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **k: criticals.append(a) or QMessageBox.StandardButton.Ok))

    def boom(path, items):
        raise TypeError("Object of type float32 is not JSON serializable")

    monkeypatch.setattr(mw, "write_full_json", boom)
    w.export_json()  # 修复前：TypeError 直接从 Qt 槽里逃出来
    assert len(criticals) == 1 and "写入失败" in criticals[0][1]
    assert "float32" in criticals[0][2]
    assert not (tmp_path / "out.json").exists()
    w.close()


def test_hover_highlights_only_when_cursor_inside_colony(qtbot, tmp_path):
    """鼠标**划过菌落附近**不该把它点亮——只有光标落在圈内才高亮。

    原来 hover 走的是带点击容差的 `_hit`（`max(r*scale+4, 10)`）：菌落只有几个像素时
    10px 的"最小容差"比菌落本身还大，鼠标离得老远就亮起一圈、状态栏 tooltip 也跟着变，
    看起来像被选中了。现在 `_hit` 只有一条判据 = `_redraw_overlay` 真正画出来的半径
    （`max(c.r*scale, 3)`），悬停/添加/删除共用；不好点中时放大视图即可。
    """
    import cv2
    import numpy as np
    from PySide6.QtCore import QSettings, QPointF
    from PySide6.QtWidgets import QApplication

    from bac_count.core.models import Colony

    img = np.full((240, 240, 3), 230, np.uint8)
    cv2.circle(img, (120, 120), 12, (60, 60, 60), -1)
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(300)  # 让检测结果先落地，之后再注入我们自己的几何
    c = w.canvas
    col = Colony(id=1, x=120.0, y=120.0, r=12.0, area=452.0)
    c.set_colonies([col], 1.0)  # 场景坐标 = 原图坐标，便于按像素算

    def move_to(sx: float, sy: float) -> None:
        qtbot.mouseMove(c.viewport(), pos=c.mapFromScene(QPointF(sx, sy)))
        QApplication.processEvents()

    move_to(120, 120)  # 圈内（圆心）
    assert c._hover is col
    assert c.toolTip().startswith("#1")

    move_to(130, 120)  # 圈内边缘（d=10 ≤ r=12）
    assert c._hover is col

    move_to(134, 120)  # 圈外 d=14：旧的 hover 容差 max(12+4,10)=16 会点亮，现在不会
    assert c._hover is None, "圈外 2px 就被选中了"
    assert c.toolTip() == ""

    move_to(145, 120)  # 更远 d=25：两种判据都不该命中
    assert c._hover is None

    # 点击与悬停同判据：圈外一律不命中（不好点中就放大视图）
    assert c._hit(134, 120) is None
    assert c._hit(130, 120) is col

    # 极小菌落：判定半径跟画出来的半径一致（_redraw_overlay 有 3px 下限），
    # 否则缩放视图下 r<1px 的菌落永远悬停不上
    tiny = Colony(id=2, x=50.0, y=50.0, r=0.5, area=1.0)
    c.set_colonies([tiny], 1.0)
    move_to(52, 50)  # d=2 ≤ 画出的 3px
    assert c._hover is tiny
    move_to(56, 50)  # d=6 > 3
    assert c._hover is None

    # 鼠标移出画布：点亮不能僵在最后一个菌落上
    c.set_colonies([col], 1.0)  # 上一步换成 tiny 后列表里已不含 col
    move_to(120, 120)
    assert c._hover is col
    from PySide6.QtCore import QEvent

    QApplication.sendEvent(c, QEvent(QEvent.Type.Leave))
    QApplication.processEvents()
    assert c._hover is None and c.toolTip() == "", "移出画布后高亮残留"
    w.close()


def test_click_requires_cursor_inside_colony(qtbot, tmp_path):
    """添加/删除的点击与悬停**同一条判据**：必须落在这只菌落的圈里。

    以前点击带 `max(r+4, 10)` 容差：菌落只有几个像素时会被"隔空"选中/删掉，
    误触完全看不出来。现在圈外一律不命中——删除模式点圈外什么都不发生；
    添加模式点圈外（哪怕只隔 2px）就是在空白处加一只。
    菌落太小不好点中时放大视图：1 个屏幕像素对应的场景距离变小，圈内落点能精确戳到。
    """
    import cv2
    import numpy as np
    from PySide6.QtCore import QSettings, QPointF, Qt
    from PySide6.QtWidgets import QApplication

    from bac_count.core.models import Colony
    from bac_count.ui.canvas import MODE_ADD, MODE_REMOVE

    img = np.full((240, 240, 3), 230, np.uint8)
    cv2.circle(img, (120, 120), 12, (60, 60, 60), -1)
    f = tmp_path / "scene.png"
    assert cv2.imwrite(str(f), img)

    w = MainWindow(settings=QSettings(str(tmp_path / "s.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([f])
    qtbot.wait(300)
    c = w.canvas
    col = Colony(id=1, x=120.0, y=120.0, r=12.0, area=452.0)

    def click(sx: float, sy: float) -> None:
        qtbot.mouseClick(c.viewport(), Qt.LeftButton, pos=c.mapFromScene(QPointF(sx, sy)))
        QApplication.processEvents()

    emitted: list = []
    c.colonyClicked.connect(lambda hit: emitted.append(hit))

    # 删除模式：圈外 2px 不命中，圈内命中并发出信号
    w._set_mode(MODE_REMOVE)
    c.set_colonies([col], 1.0)
    click(134, 120)
    assert emitted == [], "圈外 2px 也能删掉菌落 → 点击没走严格判据"
    click(120, 120)
    assert emitted == [col]

    # 添加模式：圈内不新增（命中即视为"点在菌落上"），圈外视为空白处新增
    w._set_mode(MODE_ADD)
    c.set_colonies([col], 1.0)  # 上一步 _refresh_merge 会把画布列表换成检测结果，重新注入
    n_add = len(w._history().state.additions)
    click(120, 120)
    assert len(w._history().state.additions) == n_add, "点在菌落上却被当成空白处新增了"
    click(134, 120)
    assert len(w._history().state.additions) == n_add + 1, "圈外应可正常新增"
    w.close()


def test_show_ids_checkbox(qtbot):
    """右上角「显示编号」单独控制菌落圈右边的数字；「显示标记」是它的上级开关。"""
    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files[:1])
    qtbot.wait(600)
    c = w.canvas
    assert c._chk_ids.isChecked() and c._show_ids
    circles = [it for it in c._overlay if it.data(0) == 0]
    ids = [it for it in c._overlay if it.data(0) == 1]
    assert circles and ids and len(circles) == len(ids)
    assert all(it.isVisible() for it in c._overlay)

    c._chk_ids.setChecked(False)  # 只剩圈，没有数字
    assert all(it.isVisible() for it in circles)
    assert all(not it.isVisible() for it in ids)
    assert c._chk_marks.isChecked() and c._chk_ids.isEnabled()

    c._chk_ids.setChecked(True)
    assert all(it.isVisible() for it in c._overlay)

    # 上级关掉：全隐 + 编号框置灰（免得勾了没反应让人以为坏了）
    c._chk_marks.setChecked(False)
    assert all(not it.isVisible() for it in c._overlay)
    assert not c._chk_ids.isEnabled()
    c._chk_marks.setChecked(True)
    assert c._chk_ids.isEnabled() and all(it.isVisible() for it in c._overlay)

    # 关着编号时来了新结果（_redraw_overlay 重建 item）也要保持关的状态
    c._chk_ids.setChecked(False)
    c.set_colonies(c._colonies, c._scale)
    circles2 = [it for it in c._overlay if it.data(0) == 0]
    ids2 = [it for it in c._overlay if it.data(0) == 1]
    assert all(it.isVisible() for it in circles2) and all(not it.isVisible() for it in ids2)
    w.close()


def test_canvas_takes_focus_on_hover(qtbot):
    """鼠标移到图片上，画布自动拿到焦点：空格/1/2/3/Esc 立刻可用，不用先点一下。

    但正在往文本框里打字时不抢焦点（面板 spinbox 的内层 QLineEdit、预设命名框等），
    否则鼠标路过就打断输入。
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication, QLineEdit

    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files[:1])
    qtbot.wait(400)
    c = w.canvas
    enter = lambda: QApplication.sendEvent(c, QEvent(QEvent.Type.Enter))  # noqa: E731

    w.file_list.setFocus()
    QApplication.processEvents()
    assert not c.hasFocus() and w.focusWidget() is w.file_list
    enter()
    QApplication.processEvents()
    assert c.hasFocus(), "鼠标进入画布没有自动取焦点"

    # 焦点已在画布 → 重复进入不报错、不丢失
    enter()
    assert c.hasFocus()

    # 正在输入文本：不抢
    box = QLineEdit(w)
    box.setObjectName("typing")
    box.show()
    box.setFocus()
    QApplication.processEvents()
    assert w.focusWidget() is box
    enter()
    QApplication.processEvents()
    assert not c.hasFocus(), "从正在输入的文本框抢走了焦点"
    assert w.focusWidget() is box

    # 非文本控件（文件列表）之间仍会正常让位给画布
    w.file_list.setFocus()
    QApplication.processEvents()
    enter()
    assert c.hasFocus()
    w.close()


def test_manual_add_radius_follows_auto_median(qtbot):
    """manual_r=0（新默认）→ 手加的菌落半径取**当前自动检出**半径的中位数。

    没有检出任何菌落时退回 6px；面板填 >0 就用字面值。中位数只该看自动结果，
    手加的不能自己参与（否则连点几下会把半径越拉越偏）。
    """
    import math

    import numpy as np
    import pytest

    from bac_count.core.models import Colony

    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files[:1])
    qtbot.wait(600)
    assert w.last_auto and w.params["manual_r"] == 0.0
    med = float(np.median([c.r for c in w.last_auto]))

    def last_manual() -> Colony:
        manual = [c for c in w._current_result().colonies if c.source == "manual"]
        assert manual, "没加上菌落"
        return manual[-1]

    w._on_add_click(10, 10)
    c0 = last_manual()
    assert c0.r == pytest.approx(med)
    assert c0.area == pytest.approx(math.pi * med * med, rel=1e-4)

    # 手加的不参与中位数：再点一次半径仍是自动结果的中位数
    w._on_add_click(20, 20)
    assert last_manual().r == pytest.approx(med)

    w.params["manual_r"] = 15.0  # 显式值优先
    w._on_add_click(30, 30)
    assert last_manual().r == 15.0

    w.params["manual_r"] = 0.0  # 没有自动检出 → 兜底 6px
    w.last_auto = []
    w._on_add_click(40, 40)
    assert last_manual().r == 6.0
    w.close()


# ---------------------------------------------------------------- 拖动排序 / 按名称排序

def _make_images(tmp_path, names):
    """造几张真实可读的小图（名字里的数字故意让"词法序 ≠ 数值序"）。"""
    import cv2
    import numpy as np

    out = []
    for n in names:
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.circle(img, (30, 30), 6, (120, 120, 120), -1)
        p = tmp_path / n
        assert cv2.imwrite(str(p), img)
        out.append(p)
    return out


def test_natural_sort_key_orders_numbers_by_value():
    from bac_count.ui.file_list import natural_sort_key

    names = ["图10.png", "图2.png", "E12-2.png", "E0-3.png",
             "plate9.png", "plate10.png", "a.png", "B.png"]
    got = sorted(names, key=natural_sort_key)
    assert got == ["a.png", "B.png", "E0-3.png", "E12-2.png",
                   "plate9.png", "plate10.png", "图2.png", "图10.png"]
    # 数字按大小，不是按字符：图2 必须在 图10 之前
    assert got.index("图2.png") < got.index("图10.png")
    assert got.index("E0-3.png") < got.index("E12-2.png")


def test_sort_by_name_menu_reorders_and_current_follows(qtbot, monkeypatch, tmp_path):
    """右键「按名称排序」：列表与 self.images 同步、当前图跟着自己走、选区按文件保留。"""
    from bac_count.ui.file_list import FileList

    captured = []
    monkeypatch.setattr(FileList, "_popup_menu", lambda self, menu, pos: captured.append(menu))
    files = _make_images(tmp_path, ["图10.png", "图2.png", "E12-2.png", "E0-3.png", "plate9.png"])
    want = ["E0-3.png", "E12-2.png", "plate9.png", "图2.png", "图10.png"]  # 数值序

    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    vp = fl.viewport()
    pos = lambda r: fl.visualItemRect(fl.item(r)).center()
    tips = lambda: [fl.item(i).toolTip() for i in range(fl.count())]
    sel_tips = lambda: {fl.item(fl.row(i)).toolTip() for i in fl.selectedItems()}

    with qtbot.waitSignal(w.service.worker.done, timeout=20000):  # 先切到 row2
        qtbot.mouseClick(vp, Qt.LeftButton, pos=pos(2))
    qtbot.wait(200)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):  # 再切到 row0 = 图10.png
        qtbot.mouseClick(vp, Qt.LeftButton, pos=pos(0))
    qtbot.wait(200)
    cur = w.images[w.idx]
    assert cur.name == "图10.png"
    qtbot.mouseClick(vp, Qt.LeftButton, Qt.ControlModifier, pos(2))
    qtbot.mouseClick(vp, Qt.LeftButton, Qt.ControlModifier, pos(4))
    qtbot.mouseClick(vp, Qt.RightButton, pos=pos(1))  # 右键加选（既有行为）
    _send_context_menu(fl, pos(1))
    act = next(a for a in captured[-1].actions() if a.text() == "按名称排序")
    assert act.isEnabled()
    sel_before = sel_tips()
    assert len(sel_before) == 4
    act.trigger()
    qtbot.wait(100)

    assert tips() == [str(tmp_path / n) for n in want], "列表行序没有按名称排"
    assert [p.name for p in w.images] == want, "self.images 没跟着列表走（导航/导出序）"
    assert w.images[w.idx] == cur, "当前图应该跟着自己走，不该跳到别的图"
    bolded = [fl.item(i).toolTip() for i in range(fl.count()) if fl.item(i).font().bold()]
    assert bolded == [str(cur)], "加粗标记没跟着当前图走"
    assert sel_tips() == sel_before, "排序销毁了选区"
    w.close()


def test_drag_reorder_syncs_images_and_blocks_accidental_switch(qtbot, monkeypatch):
    """模拟 Qt 内部移动（offscreen 平台没有真实 DnD，只能替掉 Qt 那半段）：
    FileList.startDrag 在真实拖动里由 Qt 自己调用 → 新顺序必须写回导航/导出序。"""
    from PySide6.QtWidgets import QListWidget

    def fake_internal_move(self, actions):  # 把按住的第 0 行搬到列表末尾
        self.insertItem(self.count() - 1, self.takeItem(0))

    monkeypatch.setattr(QListWidget, "startDrag", fake_internal_move)
    files = sorted(SAMPLE_DIR.glob("*.png"))
    assert len(files) >= 3
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    vp = fl.viewport()
    pos = lambda r: fl.visualItemRect(fl.item(r)).center()

    with qtbot.waitSignal(w.service.worker.done, timeout=20000):  # 当前图 = files[1]
        qtbot.mouseClick(vp, Qt.LeftButton, pos=pos(1))
    qtbot.wait(200)
    cur = w.images[w.idx]

    qtbot.mousePress(vp, Qt.LeftButton, pos=pos(0))  # 按住第 0 行（还没松手，不该切图）
    fl.startDrag(Qt.MoveAction)                      # Qt 真实拖动时自己会调这一句
    qtbot.wait(100)

    expected = files[1:] + files[:1]
    assert [str(p) for p in w.images] == [str(p) for p in expected], "拖动结果没写回 self.images"
    assert w.images[w.idx] == cur, "当前图应该跟着自己走"
    assert w.idx == 0
    tips = [fl.item(i).toolTip() for i in range(fl.count())]
    assert tips == [str(p) for p in expected]
    assert fl.item([str(p) for p in expected].index(str(cur))).font().bold(), "加粗标记没跟着走"
    assert fl.currentRow() == w.idx, "列表 current 行没跟上（Enter/方向键会跳到别的图）"

    # 拖动结束后 Qt 还会送一次 release：不能被当成"单击切图"
    qtbot.mouseRelease(vp, Qt.LeftButton, pos=pos(fl.count() - 1))
    qtbot.wait(100)
    assert w.idx == 0 and w.images[w.idx] == cur, "拖完序被误当成单击切图"
    w.close()


def test_cancelled_drag_changes_nothing(qtbot, monkeypatch):
    """拖到不接收它的目标（画布只认图片 url）→ Qt 不动条目，我们也绝不改顺序、不误切图。"""
    from PySide6.QtWidgets import QListWidget

    monkeypatch.setattr(QListWidget, "startDrag", lambda self, actions: None)
    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    vp = fl.viewport()
    pos = lambda r: fl.visualItemRect(fl.item(r)).center()
    ordered = []
    fl.orderChanged.connect(ordered.append)
    before = [str(p) for p in w.images]

    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        qtbot.mouseClick(vp, Qt.LeftButton, pos=pos(1))
    qtbot.wait(200)
    qtbot.mousePress(vp, Qt.LeftButton, pos=pos(3))
    fl.startDrag(Qt.MoveAction)
    qtbot.mouseRelease(vp, Qt.LeftButton, pos=pos(3))
    qtbot.wait(100)
    assert not ordered, "没有移动却广播了新顺序"
    assert [str(p) for p in w.images] == before, "顺序被凭空改动"
    assert w.idx == 1 and w.images[w.idx] == files[1], "取消的拖动被误当成单击切图"
    w.close()


def test_internal_drag_mime_is_never_an_import(qtbot):
    """内部拖动的 mime 不含 url → 决不能被当成"拖入图片导入"。"""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDropEvent

    from bac_count.ui.file_list import FileList

    class _Ev:  # 只提供 mimeData()
        def __init__(self, m):
            self._m = m

        def mimeData(self):
            return self._m

    fl = FileList()
    qtbot.addWidget(fl)
    paths = [Path("/tmp/a1.png"), Path("/tmp/a10.png"), Path("/tmp/b2.png")]
    fl.set_paths(paths)
    fl.resize(200, 160)
    fl.show()
    qtbot.wait(50)
    dropped, ordered = [], []
    fl.pathsDropped.connect(dropped.append)
    fl.orderChanged.connect(ordered.append)

    mime = fl.model().mimeData([fl.model().index(0, 0)])
    assert not mime.hasUrls(), "内部拖动的 mime 现在带 url 了 —— 会和拖入导入撞车"
    assert not fl._mime_ok(_Ev(mime))
    fl.dropEvent(QDropEvent(QPoint(10, 30), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier))
    assert not dropped, "内部拖动被当成了图片导入"
    assert fl.count() == 3 and [str(p) for p in fl.pathsInOrder()] == [str(p) for p in paths]
    assert not ordered, "Qt 没有搬条目，我们也不该凭空改顺序"


def test_order_guard_rebuilds_list_when_entries_disagree(qtbot):
    """列表条目数与 self.images 对不上时：以 self.images 为准重建，导航/导出不会错行。"""
    files = sorted(SAMPLE_DIR.glob("*.png"))
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths(files)
    qtbot.wait(300)
    fl = w.file_list
    fl.orderChanged.emit([w.images[0]])  # 模拟条目丢失
    qtbot.wait(100)
    assert [str(p) for p in w.images] == [str(f) for f in files], "self.images 被污染"
    assert [fl.item(i).toolTip() for i in range(fl.count())] == [str(f) for f in files], "列表没有重建"
    w.close()
