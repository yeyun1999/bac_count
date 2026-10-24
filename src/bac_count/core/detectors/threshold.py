"""① 全局阈值 + 轮廓。"""
from __future__ import annotations

from typing import Any

import cv2

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.pipeline import contours_to_colonies
from bac_count.core.registry import Detector, PrepContext, register


@register
class GlobalThreshold(Detector):
    name = "threshold"
    display = "全局阈值+轮廓"

    def schema(self) -> list[ParamSpec]:
        return [
            ParamSpec("use_otsu", "Otsu 自动阈值", "bool", True, group="算法参数",
                      desc="自动挑选全局阈值（对双峰分布最有效）。低对比度菌落建议改用自适应阈值算法。"),
            ParamSpec("thresh", "阈值", "int", 128, group="算法参数", min=1, max=254, step=1,
                      desc="比该值暗的像素算菌落（极性统一后菌落恒为暗色）。关 Otsu 时生效，"
                           "在“预处理”视图看菌落/背景灰度再定。"),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        flag = cv2.THRESH_BINARY_INV | (cv2.THRESH_OTSU if params.get("use_otsu", True) else 0)
        _, binary = cv2.threshold(ctx.gray, float(params.get("thresh", 128)), 255, flag)
        if ctx.dish_mask is not None:
            binary = cv2.bitwise_and(binary, ctx.dish_mask)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_to_colonies(cnts)
