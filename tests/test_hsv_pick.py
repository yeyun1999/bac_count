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


# ---------------------------------------------------------------- 逐色独立检测 ----

A_RGB = (60, 70, 90)      # rgb：深色菌落（HSV≈(50,85,90)）
B_RGB = (120, 140, 150)   # rgb：浅色菌落 + 一条"皿缘带"（HSV≈(40,51,150)）
MED_RGB = (210, 190, 180)  # rgb：培养基（HSV≈(10,36,210)，两色窗口都不该罩住它）


def _two_color_scene() -> np.ndarray:
    """A 色菌落 4 个（其中 1 个压在 B 色长带上）+ B 色菌落 2 个。"""
    img = np.zeros((400, 400, 3), np.uint8)
    img[:, :] = _bgr(MED_RGB)
    cv2.rectangle(img, (54, 20), (66, 380), _bgr(B_RGB), -1)      # 细长"皿缘带" ≈4300px²
    for x, y in [(60, 200), (200, 90), (300, 200), (150, 300)]:   # A 色，第一个压在带上
        cv2.circle(img, (x, y), 8, _bgr(A_RGB), -1)
    for x, y in [(250, 120), (320, 320)]:
        cv2.circle(img, (x, y), 8, _bgr(B_RGB), -1)
    return img


def _hsv(colors, per):
    p = default_params("hsv")
    p.update(hsv_colors=[list(c) for c in colors], hsv_per_color=per)
    return p


def test_hsv_per_color_default_off_and_single_color_equivalent():
    """默认关；单色时两条路径必须逐值等价（不改变现有行为）。"""
    assert default_params("hsv")["hsv_per_color"] is False
    img = _two_color_scene()
    for colors in ([A_RGB], [B_RGB]):
        n_off = len(run_detection(img, None, _hsv(colors, False))[0])
        n_on = len(run_detection(img, None, _hsv(colors, True))[0])
        assert n_off == n_on, f"单色 {colors} 两模式结果应相同：{n_off} vs {n_on}"


def test_hsv_per_color_recovers_colony_swallowed_by_band():
    """并集模式下压在带上的 A 菌落与带粘成一块被 max_area 丢弃；逐色模式救回。"""
    img = _two_color_scene()
    off = run_detection(img, None, _hsv([A_RGB, B_RGB], False))[0]
    on = run_detection(img, None, _hsv([A_RGB, B_RGB], True))[0]
    assert len(on) == len(off) + 1, f"逐色应比并集多 1 个：{len(off)} → {len(on)}"
    # 救回的正是压带那个 (60,200)
    assert any(np.hypot(c.x - 60, c.y - 200) < 6 for c in on)
    assert not any(np.hypot(c.x - 60, c.y - 200) < 6 for c in off)
    # 其余菌落不受影响
    for x, y in [(200, 90), (300, 200), (150, 300), (250, 120), (320, 320)]:
        assert any(np.hypot(c.x - x, c.y - y) < 6 for c in on), f"({x},{y}) 丢了"


def test_hsv_per_color_no_double_count_same_colony():
    """同一个颜色重复取两次 → 逐色合并必须收成 1 个，不重复计数。"""
    img = _two_color_scene()
    single = len(run_detection(img, None, _hsv([A_RGB], True))[0])
    dup = len(run_detection(img, None, _hsv([A_RGB, A_RGB], True))[0])
    assert dup == single, f"重复取色不该翻倍：{single} vs {dup}"


def test_hsv_per_color_two_colors_scene_matches_union_when_disjoint():
    """两色互不重叠时（红/绿分处上下），逐色与并集结果一致。"""
    img = _scene()
    off = len(run_detection(img, None, _hsv([RED, GREEN], False))[0])
    on = len(run_detection(img, None, _hsv([RED, GREEN], True))[0])
    assert off == on == 5, f"并集 {off} / 逐色 {on}，期望都是 5"
