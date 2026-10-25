"""⑤ HSV 颜色分割 + 形态学（取色模式：🎨 采菌落色 + 容差，不再手调 6 个数值）。

与其他算法的本质区别：直接在**原色图**上做区域分割，能找到灰度无反差
（菌落与培养基同明暗）但颜色有反差的菌落——颜色过滤只能"删"，这里能"找"。
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.pipeline import contours_to_colonies
from bac_count.core.registry import Detector, PrepContext, register


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
            ParamSpec("morph_k", "形态学核", "int", 3, group="检测算法", min=0, max=9, step=2,
                      desc="先开运算去椒盐噪点、再闭运算补小孔。0=关闭；核大于菌落会直接抹掉菌落。"),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        colors = params.get("hsv_colors") or []
        if not colors:
            return []
        hsv = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2HSV)
        tol = float(params.get("hsv_tol", 40.0))
        dh = int(round(tol * 0.4))
        ds = dv = int(round(tol))
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for rgb in colors:
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
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        if ctx.dish_mask is not None:
            mask = cv2.bitwise_and(mask, ctx.dish_mask)
        k = int(params.get("morph_k", 3))
        if k >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_to_colonies(cnts)
