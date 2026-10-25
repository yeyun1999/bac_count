"""重叠/包含去重测试。"""
from __future__ import annotations

import math

from bac_count.core.dedup import circle_overlap_ratio, dedup_colonies
from bac_count.core.models import Colony


def c(x, y, r):
    return Colony(x=x, y=y, r=r, area=math.pi * r * r)


def test_contained_small_kept_big():
    big, small = c(50, 50, 20), c(52, 51, 6)
    out = dedup_colonies([big, small], 0.6)
    assert len(out) == 1 and out[0].r == 20


def test_heavy_overlap_merged():
    a, b = c(50, 50, 10), c(54, 50, 10)  # 中心距4 → 重叠比≈0.81
    assert circle_overlap_ratio(a, b) > 0.6
    assert len(dedup_colonies([a, b], 0.6)) == 1


def test_touching_kept_both():
    a, b = c(50, 50, 10), c(68, 50, 10)  # 中心距18 → 重叠比≈0.1
    assert circle_overlap_ratio(a, b) < 0.2
    assert len(dedup_colonies([a, b], 0.6)) == 2


def test_half_overlap_boundary():
    a, b = c(50, 50, 10), c(60, 50, 10)  # 中心距=半径 → 重叠比≈0.39 < 0.6
    assert len(dedup_colonies([a, b], 0.6)) == 2
    assert len(dedup_colonies([a, b], 0.35)) == 1  # 阈值调小才合并


def test_chain_greedy():
    """三个几乎同心的圈只留最大的。"""
    trio = [c(50, 50, 10), c(50, 50, 7), c(51, 50, 9)]
    out = dedup_colonies(trio, 0.6)
    assert len(out) == 1 and out[0].r == 10


def test_pipeline_dedup_default_on():
    from bac_count.core.pipeline import default_params
    p = default_params("adaptive")
    assert p["dedup_on"] is True and p["dedup_ovl"] == 0.6
