"""④ Hough 圆检测。"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.registry import Detector, PrepContext, register


@register
class HoughCircles(Detector):
    name = "hough"
    display = "Hough 圆"

    def schema(self) -> list[ParamSpec]:
        return [
            ParamSpec("h_param1", "Canny 高阈值", "int", 100, group="检测算法", min=20, max=300, step=5,
                      desc="边缘检测强度：调大只剩清晰边缘（漏检多、假阳性少）。"),
            ParamSpec("h_param2", "累加器阈值", "int", 16, group="检测算法", min=5, max=100, step=1,
                      desc="圆投票门槛：调小检出更多圆但假阳性上升。"),
            ParamSpec("min_r", "最小半径", "int", 3, group="检测算法", min=1, max=100, step=1,
                      desc="按“预处理”视图量一个典型菌落的像素半径来设区间。"),
            ParamSpec("max_r", "最大半径", "int", 20, group="检测算法", min=2, max=200, step=1,
                      desc="半径上限过大时，手写字的弧形笔画常被误判成大圆。"),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        img = cv2.GaussianBlur(ctx.gray, (3, 3), 0)
        circles = cv2.HoughCircles(
            img, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(3, int(params.get("min_r", 3))),
            param1=float(params.get("h_param1", 100)),
            param2=float(params.get("h_param2", 16)),
            minRadius=int(params.get("min_r", 3)),
            maxRadius=int(params.get("max_r", 20)),
        )
        out: list[Colony] = []
        if circles is None:
            return out
        h, w = ctx.dish_mask.shape if ctx.dish_mask is not None else img.shape
        for cx, cy, r in circles[0]:
            xi, yi = int(round(cx)), int(round(cy))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            if ctx.dish_mask is not None and ctx.dish_mask[yi, xi] == 0:
                continue
            out.append(Colony(x=cx, y=cy, r=r, area=math.pi * r * r))
        return out
