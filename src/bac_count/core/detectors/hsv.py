"""⑤ HSV 颜色分割 + 形态学（有色菌落 vs 黑马克笔/背景）。"""
from __future__ import annotations

from typing import Any

import cv2

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.pipeline import contours_to_colonies
from bac_count.core.registry import Detector, PrepContext, register


@register
class HsvSegment(Detector):
    name = "hsv"
    display = "HSV 颜色分割"

    def schema(self) -> list[ParamSpec]:
        return [
            ParamSpec("h_lo", "H 下限", "int", 10, group="算法参数", min=0, max=179, step=1),
            ParamSpec("h_hi", "H 上限", "int", 40, group="算法参数", min=0, max=179, step=1),
            ParamSpec("s_lo", "S 下限", "int", 62, group="算法参数", min=0, max=255, step=1),
            ParamSpec("s_hi", "S 上限", "int", 190, group="算法参数", min=0, max=255, step=1),
            ParamSpec("v_lo", "V 下限", "int", 85, group="算法参数", min=0, max=255, step=1),
            ParamSpec("v_hi", "V 上限", "int", 175, group="算法参数", min=0, max=255, step=1),
            ParamSpec("morph_k", "形态学核", "int", 3, group="算法参数", min=0, max=9, step=2),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        hsv = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2HSV)
        lo = (int(params["h_lo"]), int(params["s_lo"]), int(params["v_lo"]))
        hi = (int(params["h_hi"]), int(params["s_hi"]), int(params["v_hi"]))
        mask = cv2.inRange(hsv, lo, hi)
        if ctx.dish_mask is not None:
            mask = cv2.bitwise_and(mask, ctx.dish_mask)
        k = int(params.get("morph_k", 3))
        if k >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_to_colonies(cnts)
