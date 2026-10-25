"""检测流水线：预处理 → 算法 → 后处理过滤；坐标统一到原图系。"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from bac_count.core.color import apply_color_filter
from bac_count.core.models import Colony, ParamSpec
from bac_count.core.preprocess import make_dish_mask, prepare
from bac_count.core.registry import PrepContext, get_detector
from bac_count.core.split import split_merged

__all__ = ["PRE_SCHEMA", "POST_SCHEMA", "run_detection", "contours_to_colonies", "default_params"]

WORKING_MAX = 1200  # 调参/预览的工作分辨率上限（长边）

# 面板分组顺序 = 检测执行顺序：皿ROI → 预处理 → 算法(+算法参数) → 颜色过滤 → 拆分 → 过滤 → 手动
PRE_SCHEMA: list[ParamSpec] = [
    ParamSpec("rim_pct", "皿缘内缩", "float", 0.12, group="皿 ROI", min=0.0, max=0.3, step=0.01,
              desc="从皿缘向内收缩的检测禁区比例，皿缘暗环易出整圈假阳性→调大；贴边菌落被漏→调小。"
                   "“预处理”视图的红圈即当前边界。"),
    ParamSpec("channel", "颜色通道", "combo", "gray", group="预处理",
              choices=["gray", "red", "green", "blue", "hsv_h", "hsv_s", "hsv_v"],
              desc="用哪个通道做检测。挑菌落与培养基反差最大的通道，逐个切换在“预处理”视图对比。"),
    ParamSpec("blur_k", "高斯模糊核", "int", 3, group="预处理", min=0, max=15, step=2,
              desc="检测前平滑：调大压掉培养基颗粒噪声，但小菌落可能被抹掉；0=关闭。"),
    ParamSpec("clahe", "CLAHE 均衡", "bool", False, group="预处理",
              desc="限制对比度的局部直方图均衡，光照不均的平板可开；配合强度参数使用。"),
    ParamSpec("clahe_clip", "CLAHE 强度", "float", 2.0, group="预处理", min=0.5, max=6.0, step=0.1,
              desc="CLAHE 增强幅度。过大噪声被一起放大。"),
    ParamSpec("shading", "阴影校正核", "int", 0, group="预处理", min=0, max=151, step=10,
              desc="减去大核中值背景，消除渐变光斑/阴影；0=关闭，常用 51~101。核须大于菌落直径。"),
    ParamSpec("polarity", "菌落极性", "combo", "dark", group="预处理",
              choices=["dark", "light"],
              desc="dark=菌落比培养基暗（常规显色）；light=菌落比培养基亮（透明/荧光菌落）。"),
    ParamSpec("algorithm", "算法选择", "combo", "adaptive", group="检测算法",
              choices=[],  # choices 由 UI 填充
              desc="阈值类适合对比明显的常规平板；Blob 快速稳健；HSV 适合有色菌落并天然排除黑字；"
                   "Hough 适合大小一致的圆菌。"),
]

POST_SCHEMA: list[ParamSpec] = [
    ParamSpec("color_on", "启用颜色过滤", "bool", False, group="颜色过滤",
              desc="按取色器采到的颜色筛掉候选：与所有“菌落颜色”相差太远的去掉，"
                   "与“排除颜色”相近的去掉。检测后、拆分前执行，对所有算法生效。"),
    ParamSpec("keep_colors", "菌落颜色", "colorlist", [], group="颜色过滤",
              desc="点“🎨”后在图上点菌落本体取色，可取多个。非空时：颜色与所有菌落色"
                   "距离大于容差的候选被去掉。留空=不启用此规则。"),
    ParamSpec("keep_tol", "菌落色容差", "float", 60.0, group="颜色过滤", min=10.0, max=200.0, step=5.0,
              desc="菌落色允许的色差（Lab 距离：同色深浅≈15，红↔绿/黑>100）。"
                   "误删真菌落→调大；假色没滤掉→调小。"),
    ParamSpec("drop_colors", "排除颜色", "colorlist", [], group="颜色过滤",
              desc="点“🎨”后在图上点培养基/马克笔字/气泡等取色，可取多个。"
                   "颜色与此表任一项接近（小于容差）的候选被去掉。"),
    ParamSpec("drop_tol", "排除色容差", "float", 35.0, group="颜色过滤", min=5.0, max=200.0, step=5.0,
              desc="判定“同色”的阈值。黑字滤不净→调大；菌落与干扰色太接近被误删→调小。"),
    ParamSpec("split_on", "拆分粘连菌落", "bool", False, group="粘连拆分",
              desc="对面积远超典型菌落的大团做距离变换+分水岭拆分（阈值/自适应/HSV 效果好；"
                   "Blob/Hough 本身按圆找，收益小）。拆出的碎片仍受面积/圆度过滤。"),
    ParamSpec("split_factor", "合并判定倍数", "float", 1.6, group="粘连拆分", min=1.2, max=4.0, step=0.1,
              desc="候选面积 > 全体中位面积×此倍数才尝试拆分。粘连漏拆→调小；"
                   "菌落大小本来就不均导致误拆→调大。"),
    ParamSpec("split_peak", "分水岭峰值比", "float", 0.55, group="粘连拆分", min=0.3, max=0.85, step=0.05,
              desc="距离变换找双核的搜索起点（自动向上搜索至 0.9）。调小更激进，可能把单个"
                   "不规则菌落拆成两半；误拆→调大。"),
    ParamSpec("min_area", "最小面积(px²)", "int", 20, group="后处理过滤", min=1, max=5000, step=1,
              desc="面积小于此值的候选丢弃（含拆分出的碎片）。调小可检出细小菌落，但噪点变多。"),
    ParamSpec("max_area", "最大面积(px²)", "int", 800, group="后处理过滤", min=10, max=50000, step=10,
              desc="面积大于此值的候选丢弃。调小可排除马克笔字；拆分在过滤前执行，大团先拆再判。"),
    ParamSpec("min_circ", "最小圆度", "float", 0.35, group="后处理过滤", min=0.0, max=1.0, step=0.05,
              desc="圆度 4πA/P²，1=正圆。调大排除马克笔细长笔画；过小会把拉丝菌落当笔画滤掉。"),
    ParamSpec("max_aspect", "最大长宽比", "float", 4.0, group="后处理过滤", min=1.0, max=20.0, step=0.5,
              desc="最小外接矩形的长宽比上限，过滤笔画状/拉丝状目标。"),
    ParamSpec("manual_r", "手动菌落半径(px)", "float", 6.0, group="手动校正", min=1.0, max=100.0, step=0.5,
              desc="“添加”模式点击空白处时新菌落的默认半径。"),
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
    """所有参数（预处理+过滤+指定算法）的默认值。列表默认值逐个拷贝，防跨图共享。"""
    p = {s.key: (list(s.default) if isinstance(s.default, list) else s.default)
         for s in PRE_SCHEMA + POST_SCHEMA}
    if algorithm:
        p["algorithm"] = algorithm
    p.update({s.key: (list(s.default) if isinstance(s.default, list) else s.default)
              for s in get_detector(p["algorithm"]).schema()})
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
    raw = apply_color_filter(ctx, raw, params)  # ① 颜色过滤（先剔除异色大块，拆分基准更干净）
    raw = split_merged(ctx, raw, params)  # ② 粘连拆分（split_on 关闭时原样返回）

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
