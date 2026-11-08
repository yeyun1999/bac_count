"""⑤ HSV 颜色分割 + 形态学（取色模式：🎨 采菌落色 + 容差，不再手调 6 个数值）。

与其他算法的本质区别：直接在**原色图**上做区域分割，能找到灰度无反差
（菌落与培养基同明暗）但颜色有反差的菌落——颜色过滤只能"删"，这里能"找"。

多色取色有两种合并方式（`hsv_per_color`）：
- 关（默认）：各色的 `inRange` 掩膜先做**像素并集**，再一次性 `findContours`。
  省算力、单色时最干净；但两色窗口在培养基/皿缘带上重叠时，菌落会被粘成大块，
  整块超 max_area 就成片丢失（P12-1 实测第二色让第一色的 13 个菌落消失）。
- 开：每个颜色**单独**走掩膜→形态学→轮廓，再按圆心距合并候选（`_merge_by_center`）。
  同一菌落被两色命中只算一个，粘连/吞并只发生在自己那一片里，能被另一色救回。
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.pipeline import contours_to_colonies
from bac_count.core.registry import Detector, PrepContext, register

# hsv_per_color 判"同一菌落被两色重复命中"的圆心距系数：d ≤ 系数 × 两半径较大者。
# 实测 P12-1：同菌落两色命中 d≈0~2px；粘连哑铃与其子落的 d≈6px、哑铃 r≈10.5
# → 0.6×10.5=6.3 刚好把哑铃判为"与子落同一目标"而淘汰它；真正相邻的两落 d=13px、
# r≈5.5 → 0.6×5.5=3.3 < 13，判为两个菌落，不会被误并。
_MERGE_DIST = 0.6


def _merge_by_center(cands: list[Colony]) -> list[Colony]:
    """跨色重复命中合并。排序先圆度后面积：哑铃圆度低 → 让位给拆得开的子落；
    同一个菌落被两色各命中一次时，取圆度更高、面积更大的那个（窗口更全、半径更准）。"""
    kept: list[Colony] = []
    for c in sorted(cands, key=lambda k: (-k.circularity, -k.area)):
        if any(np.hypot(c.x - k.x, c.y - k.y) <= _MERGE_DIST * max(c.r, k.r) for k in kept):
            continue
        kept.append(c)
    return kept


@register
class HsvSegment(Detector):
    name = "hsv"
    display = "HSV 颜色分割"

    def schema(self) -> list[ParamSpec]:
        return [
            ParamSpec("hsv_colors", "菌落取色", "colorlist", [], group="检测算法",
                      desc="点🎨在菌落上采 1~3 个颜色即可（可 Ctrl+左键连取）。"
                           "留空 = 什么都不检测。"),
            ParamSpec("hsv_tol", "颜色容差", "float", 40.0, group="检测算法", min=5, max=120, step=5,
                      desc="取色窗口的扩展幅度：H ±容差×0.4（色相更敏感）；S 向更高饱和放、"
                           "V 向更暗放（菌落通常比培养基饱和且偏暗，向培养基方向只放一半）。"
                           "漏检同类菌→调大；圈进培养基→调小。"),
            ParamSpec("hsv_per_color", "逐色独立检测", "bool", False, group="检测算法",
                      desc="默认关：各取色的掩膜先做**像素并集**、再一次性提轮廓。两色窗口在培养基/"
                           "皿缘带上重叠时，菌落会被粘成大块，整块超 max_area 就成片丢失（实测 P12-1 "
                           "加第二色后第一色的 13 个菌落消失）。打开后每个颜色**单独提轮廓**，再按圆心距"
                           "合并候选：同一菌落被两色命中只算一个（优先保留圆度高的，粘连成的哑铃会被淘汰），"
                           "被皿缘带/邻居粘连拖累的菌落可由另一色的干净掩膜救回。采了 2 个以上颜色建议开。"),
            ParamSpec("morph_k", "形态学核", "int", 3, group="检测算法", min=0, max=9, step=2,
                      desc="先开运算去椒盐噪点、再闭运算补小孔。0=关闭；核大于菌落会直接抹掉菌落。"
                           "逐色独立检测时按每个颜色分别执行。"),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        colors = params.get("hsv_colors") or []
        if not colors:
            return []
        hsv = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2HSV)
        tol = float(params.get("hsv_tol", 40.0))
        dh = int(round(tol * 0.4))
        ds = dv = int(round(tol))
        k = int(params.get("morph_k", 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1)) if k >= 1 else None
        per_color = bool(params.get("hsv_per_color", False))
        mask = np.zeros(hsv.shape[:2], np.uint8)
        out: list[Colony] = []
        for rgb in colors:
            m = self._color_mask(hsv, rgb, dh, ds, dv)
            if per_color:
                out.extend(self._colonies_from(m, ctx, kernel))  # 每色单独提轮廓，互不污染
            else:
                mask = cv2.bitwise_or(mask, m)  # 像素级并集：轮廓最后一次性提
        if not per_color:
            return self._colonies_from(mask, ctx, kernel)
        return _merge_by_center(out)

    @staticmethod
    def _color_mask(hsv: np.ndarray, rgb, dh: int, ds: int, dv: int) -> np.ndarray:
        """单个取色的 HSV 窗口掩膜（含色相回绕拆段）。"""
        b, g, r = int(rgb[2]), int(rgb[1]), int(rgb[0])
        px = cv2.cvtColor(np.array([[[b, g, r]]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = int(px[0]), int(px[1]), int(px[2])  # OpenCV5 inRange 拒收 np 标量元组
        # 非对称窗口：菌落通常比培养基"更饱和、更暗"（实测 E0-3 菌落 S95/V142 vs
        # 培养基 S60/V170，仅差 35~40）。向培养基方向收半格、反方向全放——
        # 对称 ±tol 到 40 就会把整盘培养基圈进来（实测踩坑）。
        slo, shi = max(0, s - ds // 2), min(255, s + ds)
        vlo, vhi = max(0, v - dv), min(255, v + dv // 2)
        if dh >= 90:  # 色相窗口覆盖整圈
            ranges = [((0, slo, vlo), (179, shi, vhi))]
        else:
            lo_h, hi_h = (h - dh) % 180, (h + dh) % 180
            if lo_h <= hi_h:
                ranges = [((lo_h, slo, vlo), (hi_h, shi, vhi))]
            else:  # 窗口跨 0/179 色相回绕 → 拆两段
                ranges = [((0, slo, vlo), (hi_h, shi, vhi)),
                          ((lo_h, slo, vlo), (179, shi, vhi))]
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
        return m

    @staticmethod
    def _colonies_from(mask: np.ndarray, ctx: PrepContext, kernel) -> list[Colony]:
        """掩膜 → 皿内裁剪 → 形态学 → 外轮廓 → 菌落（工作图坐标）。"""
        if ctx.dish_mask is not None:
            mask = cv2.bitwise_and(mask, ctx.dish_mask)
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_to_colonies(cnts)
