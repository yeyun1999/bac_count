"""§1.4 尺度无关层：面板参数的口径是「原图像素」，检测在 ≤WORKING_MAX 的工作图上跑。

护栏针对的是"参数单位不一致"：面积类一直在折算（×scale²），而 block/blur_k/shading/
morph_k/min_r/max_r 这些长度类以前**不折算**——同一套参数在 >1200px 的图上其实作用在
缩过小的图上，等效强度被放大（实测同一平板放大 2.5× 后 adaptive 109→58）。

注意：折算解决的是"参数单位一致"，**不等于跨分辨率计数一致**——把图放大 2.5 倍，
菌落的 px² 面积真的变成 6.25 倍，min_area/max_area、min_r/max_r 这些绝对像素参数按
定义就该把大菌落判成"超出上限"。要做到分辨率无关得把参数钉在皿直径上归一化
（优化指南 §1.4 方案 2，属产品决策，未做）。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from bac_count.core.pipeline import WORKING_MAX, _to_working_params, default_params, run_detection
from bac_count.core.registry import DETECTORS


# ------------------------------------------------------------------ 单位换算本身
def test_scale_one_is_identity():
    """scale=1（≤WORKING_MAX 的图）下每个参数逐值不变——现有基准全部走这条路。"""
    p = default_params("hough")
    p.update(block=41, shading=51, blur_k=5, min_area=30, max_aspect=3.0)
    assert _to_working_params(p, 1.0) == p


def test_length_params_scale_with_floor():
    p = default_params("hough")  # block 不在 hough schema 里 → 用 adaptive 的默认值补
    p.update(block=31, blur_k=3, shading=0, min_r=3, max_r=20, morph_k=3)
    wp = _to_working_params(p, 0.8)
    assert wp["block"] == 25          # 31 × 0.8
    assert wp["blur_k"] == 2          # 3 × 0.8 → 2（prepare 里再 |1 → 3）
    assert wp["shading"] == 0         # 0 = 关闭，折算后仍然是关闭
    assert wp["min_r"] == 2 and wp["max_r"] == 16
    assert wp["morph_k"] == 2


def test_area_params_scale_squared():
    wp = _to_working_params(default_params("adaptive"), 0.5)
    assert wp["min_area"] == 5.0 and wp["max_area"] == 200.0  # ×0.25


def test_scale_invariant_params_untouched():
    """圆度/长宽比/颜色容差/比例类参数与像素尺度无关，折算反而会误伤。"""
    p = default_params("adaptive")
    p.update(min_circ=0.5, max_aspect=3.0, keep_tol=60.0, dedup_ovl=0.7, rim_pct=0.2, C=7.0)
    wp = _to_working_params(p, 0.4)
    for k in ("min_circ", "max_aspect", "keep_tol", "dedup_ovl", "rim_pct", "C"):
        assert wp[k] == p[k], k


def test_radius_range_never_collapses():
    p = default_params("hough")
    p.update(min_r=20, max_r=21)
    wp = _to_working_params(p, 0.1)  # 20→2、21→2，区间塌了
    assert wp["max_r"] > wp["min_r"] >= 1


# ------------------------------------------------------------------ 折算真的传到了 cv2
def _capture(call_log: list, real):
    def wrapper(*a, **kw):
        call_log.append((a, kw))
        return real(*a, **kw)
    return wrapper


def test_adaptive_block_reaches_cv2_scaled(plate, monkeypatch):
    """放大到 >WORKING_MAX 后，传给 adaptiveThreshold 的 blockSize 必须是折算值。"""
    bgr, alpha, _ = plate
    big = cv2.resize(bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    bal = cv2.resize(alpha, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
    scale = WORKING_MAX / big.shape[0]
    assert scale < 1.0, "构造的图必须触发缩放分支（scale<1 的回归网以前是空白）"

    calls: list = []
    real = cv2.adaptiveThreshold
    monkeypatch.setattr(cv2, "adaptiveThreshold", _capture(calls, real))
    params = default_params("adaptive")
    run_detection(big, bal, params)
    blocks = [a[4] for a, _kw in calls if len(a) >= 5]  # (src,255,method,type,blockSize,C)
    assert blocks, "adaptiveThreshold 没被调用"
    expected = max(3, int(round(params["block"] * scale))) | 1
    assert expected in blocks, f"blockSize {blocks} 里没有折算后的 {expected}（说明长度参数没折算）"
    assert 31 not in blocks, "原图口径的 block=31 被原样用在了缩放后的工作图上"


def test_hough_radii_reach_cv2_scaled(plate, monkeypatch):
    """Hough 的 minRadius/maxRadius 同样要按 scale 折算（否则原图调好的半径区间
    在工作图上等效放大，检出偏大偏少的圆）。"""
    b, a, _truth = plate
    big = cv2.resize(b, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    bal = cv2.resize(a, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
    scale = WORKING_MAX / big.shape[0]

    calls: list = []
    real = cv2.HoughCircles
    monkeypatch.setattr(cv2, "HoughCircles", _capture(calls, real))
    cols = run_detection(big, bal, default_params("hough"))[0]
    assert calls, "HoughCircles 没被调用"
    # 菌落检测那次调用的关键字参数里带 minRadius/maxRadius（找皿那次用的是模块常量）
    radii = [kw for _a, kw in calls if "minRadius" in kw]
    assert radii, f"Hough 调用没带 minRadius: {calls}"
    kw = radii[0]
    assert kw["minRadius"] == max(1, int(round(3 * scale)))
    assert kw["maxRadius"] == max(kw["minRadius"] + 1, int(round(20 * scale)))
    assert kw["maxRadius"] != 20, "半径区间没折算"
    assert isinstance(cols, list)


# ------------------------------------------------------------------ scale<1 路径的基本覆盖
@pytest.mark.parametrize("algo", sorted(DETECTORS))
def test_all_algos_on_downscaled_plate(plate, algo):
    """>WORKING_MAX 的图（scale<1）：所有算法跑通且检出的菌落数落在合理区间。

    这条不是精度基准，是 §5.1 指出的覆盖空白——以前 scale<1 这条路一行测试都没有，
    参数单位折算的改动全凭肉眼。数量上下界用同一片平板的 scale=1 结果做参照。
    """
    bgr, alpha, truth = plate
    params = default_params(algo)
    if algo == "hsv":  # 取一片菌落的中心色当取色参数，否则空表 = 什么都不检测
        tx, ty, _tr = truth[0]
        params["hsv_colors"] = [list(int(v) for v in reversed(bgr[ty, tx]))]
        params["color_on"] = True
    base = len(run_detection(bgr, alpha, params)[0])
    big = cv2.resize(bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    bal = cv2.resize(alpha, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
    cols, ctx, scale = run_detection(big, bal, params)
    assert scale < 1.0
    assert ctx.bgr.shape[0] <= WORKING_MAX
    if base == 0:
        pytest.skip("参照结果为空，区间无意义")
    # 放大 2.5× 后菌落的原图 px² 面积变大 → 按定义会有一批超出 max_area，
    # 检出数只允许下降、不允许归零或爆炸（归零=参数被缩到静默失效的早期信号）。
    assert 0 < len(cols) <= base * 1.2, f"{algo}: scale=1 时 {base} 个，放大后 {len(cols)} 个"
    # 坐标/尺寸必须回到原图口径（原图 1500px，菌落半径不该还是工作图的量级）
    r_med = float(np.median([c.r for c in cols]))
    assert r_med > 4.0, f"半径中位数 {r_med} 没换算回原图尺度"
