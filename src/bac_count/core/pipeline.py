"""检测流水线：预处理 → 算法 → 后处理过滤；坐标统一到原图系。"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from bac_count.core.models import Colony, ParamSpec
from bac_count.core.preprocess import make_dish_mask, prepare
from bac_count.core.registry import PrepContext, get_detector

__all__ = ["PRE_SCHEMA", "POST_SCHEMA", "run_detection", "contours_to_colonies", "default_params"]

WORKING_MAX = 1200  # 调参/预览的工作分辨率上限（长边）

PRE_SCHEMA: list[ParamSpec] = [
    ParamSpec("algorithm", "检测算法", "combo", "adaptive", group="算法",
              choices=[]),  # choices 由 UI 填充
    ParamSpec("channel", "颜色通道", "combo", "gray", group="预处理",
              choices=["gray", "red", "green", "blue", "hsv_h", "hsv_s", "hsv_v"]),
    ParamSpec("blur_k", "高斯模糊核", "int", 3, group="预处理", min=0, max=15, step=2),
    ParamSpec("clahe", "CLAHE 均衡", "bool", False, group="预处理"),
    ParamSpec("clahe_clip", "CLAHE 强度", "float", 2.0, group="预处理", min=0.5, max=6.0, step=0.1),
    ParamSpec("shading", "阴影校正核", "int", 0, group="预处理", min=0, max=151, step=10),
    ParamSpec("polarity", "菌落极性", "combo", "dark", group="预处理",
              choices=["dark", "light"]),
    ParamSpec("rim_pct", "皿缘内缩", "float", 0.12, group="预处理", min=0.0, max=0.3, step=0.01),
]

POST_SCHEMA: list[ParamSpec] = [
    ParamSpec("min_area", "最小面积(px²)", "int", 20, group="过滤", min=1, max=5000, step=1),
    ParamSpec("max_area", "最大面积(px²)", "int", 800, group="过滤", min=10, max=50000, step=10),
    ParamSpec("min_circ", "最小圆度", "float", 0.35, group="过滤", min=0.0, max=1.0, step=0.05),
    ParamSpec("max_aspect", "最大长宽比", "float", 4.0, group="过滤", min=1.0, max=20.0, step=0.5),
]


def contours_to_colonies(contours, scale: float = 1.0) -> list[Colony]:
    """轮廓 → 菌落（工作图坐标）。circularity/aspect 尺度无关。"""
    out: list[Colony] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        per = cv2.arcLength(cnt, True)
        circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        aspect = max(w, h) / min(w, h) if min(w, h) > 0.5 else 99.0
        out.append(Colony(x=cx, y=cy, r=r, area=area, circularity=circ, aspect=aspect))
    return out


def _post_filter(colonies: list[Colony], params: dict[str, Any], dish_mask: np.ndarray | None) -> list[Colony]:
    amin = float(params.get("min_area", 20))
    amax = float(params.get("max_area", 800))
    cmin = float(params.get("min_circ", 0.35))
    amax_aspect = float(params.get("max_aspect", 4.0))
    out = []
    for c in colonies:
        if not (amin <= c.area <= amax):
            continue
        if c.circularity < cmin or c.aspect > amax_aspect:
            continue
        if dish_mask is not None:
            h, w = dish_mask.shape
            xi, yi = int(round(c.x)), int(round(c.y))
            if not (0 <= xi < w and 0 <= yi < h and dish_mask[yi, xi] > 0):
                continue
        out.append(c)
    return out


def default_params(algorithm: str | None = None) -> dict[str, Any]:
    """所有参数（预处理+过滤+指定算法）的默认值。"""
    p = {s.key: s.default for s in PRE_SCHEMA + POST_SCHEMA}
    if algorithm:
        p["algorithm"] = algorithm
    p.update({s.key: s.default for s in get_detector(p["algorithm"]).schema()})
    return p


def run_detection(
    bgr: np.ndarray,
    alpha: np.ndarray | None,
    params: dict[str, Any],
    work_max: int = WORKING_MAX,
) -> tuple[list[Colony], PrepContext, float]:
    """在原图上跑检测；内部缩放到工作分辨率。返回菌落为**原图坐标**。

    返回 (colonies, ctx, scale)：scale = 工作/原图 比例，UI 用工作坐标时乘 scale。
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, work_max / max(h, w))
    if scale < 1.0:
        wb = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        wa = cv2.resize(alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if alpha is not None else None
    else:
        wb, wa = bgr, alpha

    dish_mask = make_dish_mask(wb, wa, float(params.get("rim_pct", 0.12)))
    gray = prepare(wb, wa, dish_mask, params)
    ctx = PrepContext(bgr=wb, gray=gray, dish_mask=dish_mask, alpha=wa)

    det = get_detector(params.get("algorithm", "adaptive"))
    raw = det.detect(ctx, params)

    # 面积参数按原图 px² 定义 → 换算到工作尺度
    p = dict(params)
    s2 = scale * scale
    p["min_area"] = float(params.get("min_area", 20)) * s2
    p["max_area"] = float(params.get("max_area", 800)) * s2
    kept = _post_filter(raw, p, dish_mask)

    inv = 1.0 / scale
    for c in kept:
        c.x *= inv
        c.y *= inv
        c.r *= inv
        c.area *= inv * inv
    return kept, ctx, scale
