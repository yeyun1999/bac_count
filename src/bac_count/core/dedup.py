"""重复菌落合并：两圆交集面积占小圆比例超阈值 → 判为同一菌落，保留大圆。

覆盖 Hough/拆分常见的两类重复：
- 大圆套小圆（菌落有深色中心时 Hough 出双圈）：小圆被完全包含，重叠比=1；
- 绝大部分重叠（圆参数抖动）：重叠比 →1。
真正相邻的两个菌落只在边界相交，重叠比很低（相切≈0，中心距=半径时≈0.1），
因此阈值 0.6 是安全区。贪心：按面积降序，后来者与已保留者比较。
"""
from __future__ import annotations

import math

from bac_count.core.models import Colony

__all__ = ["dedup_colonies", "circle_overlap_ratio"]


def _intersect_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    c1 = (d * d + r1 * r1 - r2 * r2) / (2 * d * r1)
    c2 = (d * d + r2 * r2 - r1 * r1) / (2 * d * r2)
    a1 = r1 * r1 * math.acos(max(-1.0, min(1.0, c1)))
    a2 = r2 * r2 * math.acos(max(-1.0, min(1.0, c2)))
    a3 = 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    return a1 + a2 - a3


def circle_overlap_ratio(a: Colony, b: Colony) -> float:
    """交集 / 较小圆面积 ∈ [0,1]。"""
    small = math.pi * min(a.r, b.r) ** 2
    if small <= 0:
        return 0.0
    d = math.hypot(a.x - b.x, a.y - b.y)
    return _intersect_area(a.r, b.r, d) / small


def dedup_colonies(colonies: list[Colony], ovl: float = 0.6) -> list[Colony]:
    if len(colonies) < 2 or ovl >= 1.0:
        return list(colonies)
    kept: list[Colony] = []
    for c in sorted(colonies, key=lambda k: -k.area):
        if any(circle_overlap_ratio(c, k) >= ovl for k in kept):
            continue
        kept.append(c)
    return kept
