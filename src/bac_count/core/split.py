"""粘连菌落拆分：对"明显过大"的候选做距离变换 + 分水岭，拆成多个子菌落。

思路（针对近圆菌落）：
1. 用自适应阈值从预处理灰度图生成前景二值图（与具体算法无关的通用"菌落=暗斑"假设）；
2. 距离变换 dt：单个圆斑的 dt 是单峰圆顶；两个粘连圆斑呈双峰+鞍部；
3. 面积 > 全体中位面积 × split_factor 的候选才尝试拆分；
4. 从 split_peak(下限) 向上搜索阈值，取第一个把峰分成 ≥2 个连通核的比例，
   以核为 marker 跑 cv2.watershed，提取子区域为 Colony；
5. 拆出的碎片仍走 pipeline 的面积/圆度过滤，宁缺毋滥。
"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.color import median_lab_at
from bac_count.core.models import Colony
from bac_count.core.registry import PrepContext

__all__ = ["split_merged"]

_PEAK_SEARCH_TOP = 0.9    # 阈值搜索上限（更高比例对噪声双峰过于宽容）
_PEAK_SEARCH_STEP = 0.05
_MIN_SUB_AREA_RATIO = 0.15  # 子区域面积至少占原候选的比例，滤掉分水岭碎屑
_FRAG_COLOR_MAX = 45.0      # 碎片颜色判定阈值（Lab）：与候选本体或任一兄弟碎片色距内即通过
                            # （兄弟规则救"半透明菌落重叠区更深"的误伤，用户 E0-1 实测）
_FRAG_OVERLAP_MIN = 0.5     # 碎片像素须有 ≥50% 落在原候选的连通轮廓内（拒绝 crop 混入的邻接异物）
_FRAG_ASPECT_MAX = 2.6      # 碎片形状：真拆出的菌落半块实测长宽比 ≤1.8；
_FRAG_FILL_MIN = 0.35       # 笔画交叉的"长条臂"外接圆巨大而面积小（fill≈0.3、asp≥3.7），
                            # 且与笔画同色同连通域——颜色/重叠校验都拦不住（用户 E0-1 二图实报），
                            # 还会因外接圆包含邻近真菌落被 dedup 吞掉（"1 处消失"根因）
_FILL_TRIGGER = 0.88        # 外接圆填充率低于此值 → 花生形，粘连嫌疑（单圆≈0.92）
_AREA_TRIGGER2 = 1.25       # 配合填充率的次级触发面积倍数（主触发 = split_factor）


def _split_binary(ctx: PrepContext, params: dict) -> np.ndarray:
    """通用前景二值图：复用 adaptive 的 block/C 参数（若面板上有）。"""
    block = int(params.get("block", 41)) | 1
    C = float(params.get("C", 8.0))
    b = cv2.adaptiveThreshold(ctx.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, block, C)
    if ctx.dish_mask is not None:
        b = cv2.bitwise_and(b, ctx.dish_mask)
    return b


def split_merged(ctx: PrepContext, colonies: list[Colony], params: dict) -> list[Colony]:
    """工作图坐标。split_on 关闭或无候选时原样返回。"""
    if not params.get("split_on", False) or len(colonies) < 2:
        return colonies
    areas = np.array([c.area for c in colonies], dtype=float)
    med = float(np.median(areas))
    thr = med * float(params.get("split_factor", 1.6))
    binary = _split_binary(ctx, params)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    out: list[Colony] = []
    for c in colonies:
        fill = c.area / (np.pi * c.r * c.r) if c.r > 0 else 1.0
        candidate = (c.area >= thr or (c.area >= _AREA_TRIGGER2 * med and fill < _FILL_TRIGGER)) \
            and c.r >= 4
        subs = _split_one(ctx, binary, dt, c, params) if candidate else None
        out.extend(subs if subs else [c])
    return out


def _find_markers(crop_dt: np.ndarray, frac_floor: float):
    """从 frac_floor 向上搜索第一个能分出 ≥2 个稳定峰的阈值。返回 (labels, keep)。"""
    top = max(_PEAK_SEARCH_TOP, frac_floor)
    for frac in np.arange(frac_floor, top + 1e-9, _PEAK_SEARCH_STEP):
        peak = (crop_dt >= crop_dt.max() * frac).astype(np.uint8)
        n, labels = cv2.connectedComponents(peak)
        if n - 1 < 2:
            continue
        sizes = np.bincount(labels.ravel())
        keep = {k for k in range(1, n) if sizes[k] >= 2}  # 过滤 <2px 的伪峰
        if len(keep) >= 2:
            return labels, keep
    return None


def _split_one(ctx, binary, dt, c: Colony, params) -> list[Colony] | None:
    h, w = binary.shape
    pad = int(c.r * 1.2) + 2
    x0, y0 = max(0, int(c.x - pad)), max(0, int(c.y - pad))
    x1, y1 = min(w, int(c.x + pad) + 1), min(h, int(c.y + pad) + 1)
    crop_b = binary[y0:y1, x0:x1]
    crop_dt = np.where(crop_b > 0, dt[y0:y1, x0:x1], 0.0)
    if crop_dt.size == 0 or float(crop_dt.max()) < 3.0:  # 太薄，峰值不可信
        return None
    found = _find_markers(crop_dt, float(params.get("split_peak", 0.55)))
    if found is None:
        return None
    labels, keep = found
    # watershed 标记：1=背景，2..=各峰，0=未知（前景非峰区）
    remap = {k: i + 2 for i, k in enumerate(sorted(keep))}
    markers = np.ones_like(labels, dtype=np.int32)
    for k, new in remap.items():
        markers[labels == k] = new
    markers[crop_b == 0] = 1
    markers[(crop_b > 0) & (labels == 0)] = 0
    if int((markers == 0).sum()) < 3:
        return None
    img = cv2.cvtColor(ctx.gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    try:
        cv2.watershed(img, markers)
    except cv2.error:
        return None
    # 原候选的连通轮廓：crop 二值图中包含候选中心的那个连通域
    # （拆分的二值图是色盲 adaptive 阈值，crop 里可能混入不相邻的笔画/杂质）
    n_cc, cc_labels = cv2.connectedComponents(crop_b, connectivity=8)
    ci = min(max(int(c.y) - y0, 0), crop_b.shape[0] - 1)
    cj = min(max(int(c.x) - x0, 0), crop_b.shape[1] - 1)
    comp_id = cc_labels[ci, cj]
    if comp_id == 0:  # 中心恰好不在前景：退化为 crop 内最大连通域
        if n_cc > 1:
            sizes = np.bincount(cc_labels.ravel())
            comp_id = int(np.argmax(sizes[1:]) + 1)
    orig_comp = (cc_labels == comp_id) if comp_id > 0 else np.zeros_like(crop_b, bool)

    cand_lab = median_lab_at(ctx.bgr, c.x, c.y, c.r)  # 候选本体参考色（中心种子聚类）
    frags: list[tuple[Colony, np.ndarray | None]] = []
    min_sub = c.area * _MIN_SUB_AREA_RATIO
    for new in remap.values():
        region = ((markers == new) & (crop_b > 0))
        region_u8 = region.astype(np.uint8) * 255
        cs, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_sub:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        per = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0.0
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        aspect = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0.5 else 99.0
        frag = Colony(x=cx + x0, y=cy + y0, r=r, area=area, circularity=circ, aspect=aspect)
        lab = median_lab_at(ctx.bgr, frag.x, frag.y, frag.r)
        # ① 形状校验：碎片必须像菌落（长条臂=笔画交叉产物，reject）
        if params.get("split_shape", True):
            fill = area / (np.pi * r * r) if r > 0 else 1.0
            if aspect > _FRAG_ASPECT_MAX or fill < _FRAG_FILL_MIN:
                continue
        # ② 重叠校验：碎片像素必须大多落在原候选连通轮廓内
        if params.get("split_overlap", True) and comp_id > 0:
            npix = int(region.sum())
            if npix == 0 or int((region & orig_comp).sum()) / npix < _FRAG_OVERLAP_MIN:
                continue
        frags.append((frag, lab))
    # ③ 颜色校验：与候选本体或任一兄弟碎片同色即通过
    #    （兄弟规则：半透明菌落重叠区颜色更深，种子色≠菌落本色，只看种子会全军覆没）
    if params.get("split_color", True) and len(frags) >= 2 and cand_lab is not None:
        def close(a, b):
            return a is not None and b is not None and float(np.linalg.norm(a - b)) <= _FRAG_COLOR_MAX
        frags = [(f, lab) for i, (f, lab) in enumerate(frags)
                 if close(lab, cand_lab) or any(j != i and close(lab, l2)
                                                for j, (_, l2) in enumerate(frags))]
    result = [f for f, _ in frags]
    return result if len(result) >= 2 else None
