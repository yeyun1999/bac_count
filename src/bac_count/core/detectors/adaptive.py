"""② 自适应阈值 + 轮廓（抗光照不均）。"""
from __future__ import annotations

from typing import Any

import cv2

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.pipeline import contours_to_colonies
from bac_count.core.registry import Detector, PrepContext, register


@register
class AdaptiveThreshold(Detector):
    name = "adaptive"
    display = "自适应阈值+轮廓"

    def schema(self) -> list[ParamSpec]:
        return [
            ParamSpec("block", "块大小", "int", 31, group="检测算法", min=3, max=101, step=2,
                      desc="局部均值取样窗口（自动取奇数）。必须大于菌落直径，否则大菌落整体被当背景漏检。"),
            ParamSpec("C", "常数 C", "float", 5.0, group="检测算法", min=-15.0, max=15.0, step=0.5,
                      desc="判定阈值 = 局部均值 − C。调大更严格（假阳性少、浅色菌落易漏）；调小反之。"),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        block = int(params.get("block", 31)) | 1
        binary = cv2.adaptiveThreshold(
            ctx.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block, float(params.get("C", 5.0)),
        )
        if ctx.dish_mask is not None:
            binary = cv2.bitwise_and(binary, ctx.dish_mask)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_to_colonies(cnts)
