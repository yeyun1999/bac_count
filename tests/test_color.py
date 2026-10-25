"""颜色过滤（core/color.py）与取色交互测试。"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.color import apply_color_filter, median_lab_at, rgb_to_lab
from bac_count.core.pipeline import default_params, run_detection
from bac_count.core.registry import PrepContext

RED = (210, 60, 60)     # rgb
GREEN = (60, 200, 60)
BG = (240, 235, 210)


def _bgr(rgb):  # rgb → bgr 三元组
    return rgb[::-1]


def _scene() -> np.ndarray:
    """3 红 + 2 绿菌落在浅黄培养基上（BGR）。"""
    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = _bgr(BG)
    for x, y in [(80, 90), (150, 60), (220, 90)]:
        cv2.circle(img, (x, y), 10, _bgr(RED), -1)
    for x, y in [(100, 210), (200, 210)]:
        cv2.circle(img, (x, y), 10, _bgr(GREEN), -1)
    return img


def _ctx(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return PrepContext(bgr=bgr, gray=gray, dish_mask=None, alpha=None)


def test_off_by_default():
    assert default_params("adaptive")["color_on"] is False
    assert default_params("adaptive")["keep_colors"] == []


def test_list_defaults_not_shared():
    a, b = default_params(), default_params()
    a["keep_colors"].append([1, 2, 3])
    assert b["keep_colors"] == []


def test_keep_colors():
    bgr = _scene()
    # run_detection 需要 BGR 输入
    cols = run_detection(bgr, None, default_params("adaptive"))[0]
    assert len(cols) == 5
    p = default_params("adaptive")
    p.update(color_on=True, keep_colors=[list(RED)], keep_tol=60.0)
    kept = run_detection(bgr, None, p)[0]
    assert len(kept) == 3, f"只应留下红色菌落，实得 {len(kept)}"


def test_drop_colors():
    bgr = _scene()
    p = default_params("adaptive")
    p.update(color_on=True, drop_colors=[list(GREEN)], drop_tol=35.0)
    kept = run_detection(bgr, None, p)[0]
    assert len(kept) == 3


def test_empty_lists_noop():
    bgr = _scene()
    p = default_params("adaptive")
    p.update(color_on=True)
    assert len(run_detection(bgr, None, p)[0]) == 5


def test_color_on_direct_unit():
    from bac_count.core.models import Colony

    bgr = _scene()
    cs = [Colony(x=80, y=90, r=10, area=300), Colony(x=100, y=210, r=10, area=300)]
    out = apply_color_filter(_ctx(bgr), cs, {
        "color_on": True, "keep_colors": [list(GREEN)], "keep_tol": 60.0})
    assert len(out) == 1 and out[0].y == 210  # 只留绿的


def test_tolerance_squeeze():
    """取色略有偏差时容差控制松紧：偏差≈14 的菌落色，容差 8 全滤掉、60 只留红的。"""
    bgr = _scene()
    RED2 = (190, 70, 70)  # 与 RED 的 Lab 距离≈14
    tight = default_params("adaptive")
    tight.update(color_on=True, keep_colors=[list(RED2)], keep_tol=8.0)
    assert len(run_detection(bgr, None, tight)[0]) == 0
    loose = default_params("adaptive")
    loose.update(color_on=True, keep_colors=[list(RED2)], keep_tol=60.0)
    assert len(run_detection(bgr, None, loose)[0]) == 3


def test_median_lab_matches_picked_color():
    lab = median_lab_at(_scene(), 80, 90, 10)
    assert lab is not None
    assert float(np.linalg.norm(lab - rgb_to_lab(RED))) < 12


def test_seed_clustering_on_stroke():
    """笔画候选：内切圆大多是培养基像素，中心种子聚类必须锚定墨色（E0-3 踩坑回归）。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = _bgr(BG)
    cv2.line(img, (30, 50), (70, 50), _bgr((35, 35, 38)), 3)  # 3px 横笔画
    lab = median_lab_at(img, 50, 50, 20)  # 0.6r=12px 圆内 ~95% 是培养基
    assert np.linalg.norm(lab - rgb_to_lab((35, 35, 38))) < 45, "种子没锚住墨色"
    assert np.linalg.norm(lab - rgb_to_lab(BG)) > 60, "不应得到培养基色"


def test_samples_smoke_color_on():
    """真实样例：任意颜色开启过滤不崩溃、只减不崩。"""
    import glob
    from pathlib import Path

    from bac_count.core.preprocess import load_image

    files = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "sample" / "*.png")))
    base = default_params("adaptive")
    p = dict(base, color_on=True, keep_colors=[[120, 110, 95]], keep_tol=90.0)
    for f in files:
        bgr, alpha = load_image(f)
        n0 = len(run_detection(bgr, alpha, base)[0])
        n1 = len(run_detection(bgr, alpha, p)[0])
        assert n1 <= n0, f"{Path(f).name}: 过滤后反而变多 {n0}→{n1}"
