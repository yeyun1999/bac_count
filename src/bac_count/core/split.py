"""粘连菌落拆分：距离变换 + 分水岭。

1. 从预处理灰度图做自适应阈值，得到通用前景二值图（菌落=暗，与所选算法无关）；
2. 距离变换 dt：单个圆斑的 dt 是单峰圆顶，两个粘连圆斑是双峰 + 中间鞍部；
3. 面积 > 全体中位面积 × split_factor 的候选才尝试拆分；
4. dt 的**局部极大值平台**就是"菌落核"（一个核=一个菌落），核的连通域即分水岭标记；
   split_peak 只用来丢掉相对太弱的核；
5. 分水岭在候选邻域内跑，盆地即为子菌落。拆不出 ≥2 个就保留原候选。
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from bac_count.core.models import Colony
from bac_count.core.registry import PrepContext

__all__ = ["split_merged"]

_MIN_SUB_RATIO = 0.15  # 子菌落面积至少占原候选的比例，滤掉分水岭碎屑


def _foreground(ctx: PrepContext, params: dict[str, Any]) -> np.ndarray:
    """通用前景二值图：复用面板上的 block/C（自适应阈值算法的参数）。

    自适应阈值只保留"比局部均值暗 C"的边带，比 block/2 大的菌落中间会被判成背景
    （空洞）→ dt 变成环状、处处是假峰。这里按外轮廓回填，得到实心前景。
    """
    block = int(params.get("block", 31)) | 1
    b = cv2.adaptiveThreshold(ctx.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, block, float(params.get("C", 5.0)))
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(b)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    if ctx.dish_mask is not None:
        filled = cv2.bitwise_and(filled, ctx.dish_mask)
    return filled


def split_merged(ctx: PrepContext, colonies: list[Colony], params: dict[str, Any]) -> list[Colony]:
    """工作图坐标。split_on 关闭或没有大团时原样返回。"""
    if not params.get("split_on", False) or len(colonies) < 2:
        return colonies
    med = float(np.median([c.area for c in colonies]))
    thr = med * float(params.get("split_factor", 1.6))
    binary = _foreground(ctx, params)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    out: list[Colony] = []
    for c in colonies:
        subs = _split_one(ctx, binary, dt, c, params) if c.area >= thr else None
        out.extend(subs if subs else [c])
    return out


def _split_one(ctx, binary, dt, c: Colony, params) -> list[Colony] | None:
    """拆一个大团。子菌落不足 2 个返回 None（调用方保留原候选）。"""
    h, w = binary.shape
    pad = int(c.r * 1.2) + 2
    x0, y0 = max(0, int(c.x - pad)), max(0, int(c.y - pad))
    x1, y1 = min(w, int(c.x + pad) + 1), min(h, int(c.y + pad) + 1)
    cb = binary[y0:y1, x0:x1]
    cd = np.where(cb > 0, dt[y0:y1, x0:x1], 0.0)
    if cd.size == 0 or float(cd.max()) < 3.0:  # 太薄，峰值不可信
        return None

    # 菌落核 = dt 的局部极大值平台（不是"全局峰值的某个比例"：粘连越紧鞍部越浅，
    # 按比例一刀切要么拆不开，要么把 N 联团只拆成 2 块）
    # 1=背景，2..=各核，0=前景未知区
    sep = max(2, int(round(0.5 * float(cd.max()))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sep + 1, 2 * sep + 1))
    core = ((cd >= cv2.dilate(cd, kernel))
            & (cd >= cd.max() * float(params.get("split_peak", 0.55)))).astype(np.uint8)
    n, lab = cv2.connectedComponents(core, connectivity=8)
    if n - 1 < 2:
        return None
    markers = lab + 1
    markers[(cb > 0) & (lab == 0)] = 0
    if int((markers == 0).sum()) < 3:
        return None
    img = cv2.cvtColor(ctx.gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    try:
        cv2.watershed(img, markers)
    except cv2.error:
        return None

    min_sub = c.area * _MIN_SUB_RATIO
    subs: list[Colony] = []
    for k in range(2, n + 1):
        region = ((markers == k) & (cb > 0)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_sub:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        per = cv2.arcLength(cnt, True)
        rw, rh = cv2.minAreaRect(cnt)[1]
        subs.append(Colony(
            x=cx + x0, y=cy + y0, r=r, area=area,
            circularity=4 * math.pi * area / (per * per) if per > 0 else 0.0,
            aspect=max(rw, rh) / min(rw, rh) if min(rw, rh) > 0.5 else 99.0,
        ))
    return subs if len(subs) >= 2 else None
