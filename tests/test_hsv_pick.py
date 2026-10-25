"""HSV 取色模式 + 算法切换参数合并（KeyError 崩溃回归）测试。"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.pipeline import default_params, run_detection

RED = (210, 60, 60)     # rgb
GREEN = (60, 200, 60)
BG = (240, 235, 210)


def _bgr(rgb):
    return rgb[::-1]


def _scene() -> np.ndarray:
    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = _bgr(BG)
    for x, y in [(80, 90), (150, 60), (220, 90)]:
        cv2.circle(img, (x, y), 10, _bgr(RED), -1)
    for x, y in [(100, 210), (200, 210)]:
        cv2.circle(img, (x, y), 10, _bgr(GREEN), -1)
    return img


def test_hsv_empty_colors_returns_zero():
    p = default_params("hsv")
    assert p["hsv_colors"] == []
    assert len(run_detection(_scene(), None, p)[0]) == 0


def test_hsv_pick_finds_only_that_color():
    p = default_params("hsv")
    p.update(hsv_colors=[list(RED)], hsv_tol=40.0)
    cols = run_detection(_scene(), None, p)[0]
    assert len(cols) == 3, f"取红色应检出 3 个红菌落，实得 {len(cols)}"
    assert all(c.y < 150 for c in cols)


def test_hsv_pick_two_colors():
    p = default_params("hsv")
    p.update(hsv_colors=[list(RED), list(GREEN)], hsv_tol=40.0)
    assert len(run_detection(_scene(), None, p)[0]) == 5


def test_hsv_hue_wrap():
    """色相窗跨 0/179 回绕的洋红色也能检出。"""
    MAGENTA = (200, 60, 190)  # rgb → cv2 H≈156? 用接近 179 的颜色
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = _bgr(BG)
    px = cv2.cvtColor(np.array([[MAGENTA[::-1]]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    cv2.circle(img, (100, 100), 12, _bgr(MAGENTA), -1)
    p = default_params("hsv")
    p.update(hsv_colors=[list(MAGENTA)], hsv_tol=50.0)
    assert len(run_detection(img, None, p)[0]) == 1, f"H={int(px[0])} 色检出失败"


def test_switch_algorithm_merges_defaults(qtbot, tmp_path):
    """GUI 切算法必须补齐该算法默认参数（HSV KeyError 崩溃根因回归）。"""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pathlib import Path

    from PySide6.QtCore import QSettings

    from bac_count.ui.main_window import MainWindow

    sample = sorted((Path(__file__).resolve().parents[1] / "sample").glob("*.png"))[0]
    w = MainWindow(settings=QSettings(str(tmp_path / "t.conf"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w.load_paths([sample])
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w._on_param_changed("algorithm", "HSV 颜色分割")
    assert "hsv_colors" in w.params and "hsv_tol" in w.params
    assert w.params["hsv_colors"] == []
    # 切走再切回 hsv 不覆盖用户已改的值
    w.params["hsv_tol"] = 77.0
    with qtbot.waitSignal(w.service.worker.done, timeout=20000):
        w._on_param_changed("algorithm", "全局阈值+轮廓")
        w._on_param_changed("algorithm", "HSV 颜色分割")
    assert w.params["hsv_tol"] == 77.0
    w.close()
