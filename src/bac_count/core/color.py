"""颜色过滤：候选级"取色保留/取色排除"，检测后、拆分前执行。

- keep_colors（菌落色）：候选中位色与所有菌落色的 Lab 距离都 > keep_tol → 丢弃
  （"相差太远的去掉"）。列表为空则此规则不生效。
- drop_colors（培养基/干扰色，如马克笔黑字、气泡）：候选中位色与任一排除色
  Lab 距离 < drop_tol → 丢弃（"和他颜色相近的去掉"）。
- 取色：以内切圆（0.6r）为域，但**用中心像素做种子**只统计与其色距 ≤ seed_tol 的
  像素——笔画状候选的中心圆里大半是培养基像素，直接取中位会得到培养基色而漏滤
  （E0-3 实测踩坑）；菌落本体中心=菌落色，聚类后仍是菌落全体像素。
- 距离空间：cv2 8bit Lab（L 0~255，a/b 128±）。经验：同色差异 <20，
  红↔绿/黑 相距 >100。

放在拆分前：马克笔字等异色大块先被剔除，拆分的中位面积基准更干净。
"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.models import Colony
from bac_count.core.registry import PrepContext

__all__ = ["apply_color_filter", "median_lab_at"]


def _to_lab(arr: np.ndarray) -> np.ndarray:
    """(...,3) uint8 BGR → (...,3) float32 Lab（cv2 8bit 标定）。"""
    b = arr.reshape(-1, 1, 3)
    lab = cv2.cvtColor(b, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return lab.astype(np.float32)


def rgb_to_lab(rgb) -> np.ndarray:
    return _to_lab(np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8))[0]


def median_lab_at(bgr: np.ndarray, x: float, y: float, r: float,
                  seed_tol: float = 45.0) -> np.ndarray | None:
    """中心种子聚类色 → Lab。越界/空返回 None（调用方应保留该候选）。

    内切圆(0.6r)像素中，只保留与中心像素 BGR 距离 ≤ seed_tol 者取中位；
    有效像素太少(<3)则退回全圆中位。
    """
    h, w = bgr.shape[:2]
    rr = max(1.0, 0.6 * r)
    x0, x1 = max(0, int(x - rr)), min(w, int(x + rr) + 1)
    y0, y1 = max(0, int(y - rr)), min(h, int(y + rr) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    crop = bgr[y0:y1, x0:x1]
    px = crop[disk] if disk.any() else crop.reshape(-1, 3)
    if len(px) > 400:  # 大候选抽稀，中位色统计上足够
        px = px[:: len(px) // 400]
    seed = bgr[min(max(int(round(y)), 0), h - 1), min(max(int(round(x)), 0), w - 1)].astype(np.int16)
    near = px[np.linalg.norm(px.astype(np.int16) - seed, axis=1) <= seed_tol]
    sel = near if len(near) >= 3 else px
    med = np.median(sel, axis=0).astype(np.uint8)  # BGR 中位
    return _to_lab(med.reshape(1, 1, 3))[0]


def apply_color_filter(ctx: PrepContext, colonies: list[Colony], params: dict) -> list[Colony]:
    """工作图坐标。color_on 关闭或两张颜色表都为空时原样返回。"""
    if not params.get("color_on", False):
        return colonies
    keep = params.get("keep_colors") or []
    drop = params.get("drop_colors") or []
    if not keep and not drop:
        return colonies
    keep_l = np.stack([rgb_to_lab(c) for c in keep]) if keep else None
    drop_l = np.stack([rgb_to_lab(c) for c in drop]) if drop else None
    keep_tol = float(params.get("keep_tol", 60.0))
    drop_tol = float(params.get("drop_tol", 35.0))
    out: list[Colony] = []
    for c in colonies:
        lab = median_lab_at(ctx.bgr, c.x, c.y, c.r)
        if lab is None:
            out.append(c)
            continue
        if keep_l is not None and float(np.min(np.linalg.norm(keep_l - lab, axis=1))) > keep_tol:
            continue
        if drop_l is not None and float(np.min(np.linalg.norm(drop_l - lab, axis=1))) < drop_tol:
            continue
        out.append(c)
    return out
