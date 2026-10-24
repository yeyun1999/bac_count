"""③ SimpleBlobDetector。"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.registry import Detector, PrepContext, register


@register
class BlobDetector(Detector):
    name = "blob"
    display = "Blob 检测"

    def schema(self) -> list[ParamSpec]:
        return [
            ParamSpec("min_thresh", "阈值下限", "int", 50, group="算法参数", min=0, max=255, step=5),
            ParamSpec("max_thresh", "阈值上限", "int", 220, group="算法参数", min=0, max=255, step=5),
            ParamSpec("min_circ_b", "圆度下限", "float", 0.3, group="算法参数", min=0.0, max=1.0, step=0.05),
            ParamSpec("min_inertia", "惯性比下限", "float", 0.2, group="算法参数", min=0.0, max=1.0, step=0.05),
        ]

    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        p = cv2.SimpleBlobDetector_Params()
        p.minThreshold = int(params.get("min_thresh", 50))
        p.maxThreshold = int(params.get("max_thresh", 220))
        p.thresholdStep = 10
        p.filterByColor = True
        p.blobColor = 0  # 预处理已统一为"菌落暗色"
        p.filterByArea = True
        p.minArea = 3
        p.maxArea = 100000
        p.filterByCircularity = True
        p.minCircularity = float(params.get("min_circ_b", 0.5))
        p.filterByConvexity = True
        p.minConvexity = 0.8
        p.filterByInertia = True
        p.minInertiaRatio = float(params.get("min_inertia", 0.4))
        det = cv2.SimpleBlobDetector_create(p)
        img = ctx.gray
        if ctx.dish_mask is not None:
            img = np.where(ctx.dish_mask > 0, img, 255).astype(np.uint8)
        kps = det.detect(img)
        return [
            Colony(x=k.pt[0], y=k.pt[1], r=k.size / 2.0, area=np.pi * (k.size / 2.0) ** 2)
            for k in kps
        ]
