"""§2.1 找皿拆成「找圆（可按图缓存）」+「内缩画掩膜（纯几何）」。

以前 `make_dish_mask` 每次拖滑杆都整个重算，无 alpha 的图（手机照片）要走
`cvtColor + medianBlur + HoughCircles`，1200px 实测 172.8ms，占 run_detection 的 71%
——而 rim_pct 和其他参数一样都不影响那个 Hough 圆，纯属白付。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from bac_count.core import pipeline
from bac_count.core.pipeline import UNSET, default_params, run_detection
from bac_count.core.preprocess import dish_mask_from_circle, find_dish_circle, make_dish_mask


def _both_paths_equal(bgr, alpha):
    for rim in (0.0, 0.12, 0.3):
        a = make_dish_mask(bgr, alpha, rim)
        b = dish_mask_from_circle(find_dish_circle(bgr, alpha), bgr.shape, rim)
        if a is None or b is None:
            assert a is None and b is None, f"rim={rim} 两条路一条找到一条没找到"
        else:
            assert np.array_equal(a, b), f"rim={rim} 拆分后的掩膜与旧实现不一致"
        assert (a is None) == (b is None)


def test_split_matches_legacy_mask_with_alpha(plate):
    bgr, alpha, _ = plate
    _both_paths_equal(bgr, alpha)


def test_split_matches_legacy_mask_without_alpha(plate):
    bgr, _alpha, _ = plate
    _both_paths_equal(bgr, None)


def test_rim_pct_only_changes_radius_not_hough():
    """rim_pct 只参与"内缩画圆"那一步：同一 circle 换 rim 不需要重新找圆。"""
    bgr = np.full((400, 400, 3), 240, np.uint8)
    alpha = np.zeros((400, 400), np.uint8)
    cv2.circle(alpha, (200, 200), 180, 255, -1)
    circle = find_dish_circle(bgr, alpha)
    assert circle is not None
    (cx, cy), cr = circle
    assert abs(cx - 200) < 3 and abs(cy - 200) < 3 and 175 < cr < 190
    r0 = dish_mask_from_circle(circle, bgr.shape, 0.0)
    r2 = dish_mask_from_circle(circle, bgr.shape, 0.2)
    assert int((r0 > 0).sum()) > int((r2 > 0).sum()) > 0
    assert np.all(r2[r0 == 0] == 0), "内缩后的掩膜必须是内缩前的子集"
    assert dish_mask_from_circle(None, bgr.shape, 0.1) is None


def test_pipeline_uses_cached_circle(plate, monkeypatch):
    """带 dish_circle 时不再调用 find_dish_circle，且产物与不带缓存时逐像素相同。"""
    bgr, alpha, _ = plate
    calls: list[int] = []
    real = pipeline.find_dish_circle

    def counting(b, a):
        calls.append(1)
        return real(b, a)

    monkeypatch.setattr(pipeline, "find_dish_circle", counting)
    params = default_params("adaptive")
    cols, ctx, _s = run_detection(bgr, alpha, params)
    assert calls == [1], "首跑应该自己找一次圆"
    assert ctx.dish_circle is not None

    calls.clear()
    cols2, ctx2, _s2 = run_detection(bgr, alpha, params, dish_circle=ctx.dish_circle)
    assert calls == [], "带了缓存还去跑 find_dish_circle"
    assert np.array_equal(ctx.dish_mask, ctx2.dish_mask)
    assert [c.id for c in cols] == [c.id for c in cols2]
    assert len(cols) == len(cols2)
    # rim_pct 改了也不该重跑找圆（这正是拖皿缘滑杆时的开销）
    cols3, ctx3, _ = run_detection(bgr, alpha, {**params, "rim_pct": 0.25},
                                  dish_circle=ctx.dish_circle)
    assert calls == []
    assert not np.array_equal(ctx.dish_mask, ctx3.dish_mask), "rim_pct 应改变掩膜"
    assert len(cols3) <= len(cols)


def test_worker_caches_circle_per_image(plate, monkeypatch):
    """DetectWorker 的单槽缓存：同一张图反复检测只找一次圆，换图才重找。"""
    pytest.importorskip("PySide6")
    from bac_count.ui.worker import DetectWorker

    bgr, alpha, _ = plate
    calls: list[int] = []
    real = pipeline.find_dish_circle

    def counting(b, a):
        calls.append(1)
        return real(b, a)

    monkeypatch.setattr(pipeline, "find_dish_circle", counting)
    got: list = []
    w = DetectWorker()
    w.done.connect(lambda seq, key, cols, prep: got.append(cols))
    params = default_params("adaptive")
    w._run(1, "a.png", bgr, alpha, params)   # 直接同步调用槽，不起线程
    w._run(2, "a.png", bgr, alpha, params)   # 拖滑杆：同一数组对象
    assert calls == [1], f"同一张图找圆 {len(calls)} 次"
    w._run(3, "b.png", bgr.copy(), alpha, params)  # 换图（不同对象）→ 必须重找
    assert len(calls) == 2
    assert len(got) == 3 and all(isinstance(c, list) for c in got)


def test_cache_holds_no_circle_for_image_without_dish(monkeypatch):
    """找不到皿圆时也要能缓存住（None 是合法值），否则每张无皿图每次都要重跑 Hough。"""
    pytest.importorskip("PySide6")
    from bac_count.ui.worker import DetectWorker

    img = np.full((300, 300, 3), 200, np.uint8)  # 纯色：既没 alpha 也没圆
    calls: list[int] = []
    real = pipeline.find_dish_circle

    def counting(b, a):
        calls.append(1)
        return real(b, a)

    monkeypatch.setattr(pipeline, "find_dish_circle", counting)
    w = DetectWorker()
    w._run(1, "x.png", img, None, default_params("adaptive"))
    w._run(2, "x.png", img, None, default_params("adaptive"))
    assert calls == [1], "无皿圆（None）没被缓存"
