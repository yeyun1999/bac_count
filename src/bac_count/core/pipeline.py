"""检测流水线：预处理 → 算法 → 后处理过滤；坐标统一到原图系。"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from bac_count.core.color import apply_color_filter
from bac_count.core.dedup import dedup_colonies
from bac_count.core.models import Colony, ParamSpec
from bac_count.core.preprocess import dish_mask_from_circle, find_dish_circle, prepare
from bac_count.core.registry import PrepContext, get_detector
from bac_count.core.split import split_merged

__all__ = ["PRE_SCHEMA", "POST_SCHEMA", "run_detection", "contours_to_colonies", "default_params",
           "UNSET", "WORKING_MAX"]

WORKING_MAX = 1200  # 调参/预览的工作分辨率上限（长边）
# ⚠ 只降不升：小于 1200px 的图按原尺寸检测（scale=1），大于的缩到 1200（scale<1）。
# 于是面板参数的口径是「**原图像素**」，长度类在 run_detection 里按 scale 折算
# （见 _LENGTH_PARAMS/_to_working_params）——但**同一块平板拍得更大时菌落的
# px² 面积也变大**，min_area/max_area、min_r/max_r 这些绝对像素参数并不会自动等效，
# 换相机/换分辨率仍需重调参数。要做真正的分辨率无关，得把参数钉在「皿直径
# = 固定像素」上归一化（优化指南 §1.4 方案 2），那会改动所有 ≤1200px 图的行为，
# 需连同 sample/ 回归基准一起重定标——属于产品决策，未做。


class _Unset:
    """哨兵：`run_detection(dish_circle=UNSET)` = 没带缓存，内部自己算。

    需要它是因为 `None` 本身是合法缓存值（"这张图确实没找到皿圆"）。
    """

    def __repr__(self) -> str:  # pragma: no cover
        return "<UNSET>"


UNSET: Any = _Unset()

# 面板上的长度/面积参数一律按**原图像素**口径定义（tooltip 写的 "px"/"px²" 就是原图口径），
# 而检测跑在缩放后的工作图（长边 ≤WORKING_MAX）上 → 不折算的话，同一套参数在
# 不同分辨率的图上不等价：同一块平板放大 2.5× 后 adaptive 109→58、hough 104→20。
# key → (0 是否表示"关闭", 折算后的下限)；下限是为了避免"开着的参数"被缩到 0 悄悄失效。
_LENGTH_PARAMS: dict[str, tuple[bool, int]] = {
    "block": (False, 3),      # 自适应阈值窗口
    "blur_k": (True, 1),      # 高斯模糊核
    "shading": (True, 3),     # 阴影校正中值核
    "morph_k": (True, 1),     # HSV 形态学核
    "min_r": (False, 1),      # Hough 半径区间
    "max_r": (False, 2),
}
_AREA_PARAMS: tuple[str, ...] = ("min_area", "max_area")  # 面积按 ×scale² 折算（连续量，不取整）


def _to_working_params(params: dict[str, Any], scale: float) -> dict[str, Any]:
    """原图口径的长度/面积参数 → 工作尺度。scale=1.0 时逐值等价（原样返回副本）。

    不折算的量都是**尺度无关**的：min_circ（4πA/P²）、max_aspect（minAreaRect 长宽比，
    实测同一平板放大 2.5× 后原始候选的中位 aspect 1.08→1.06，本来就与尺度无关）、
    Lab 色差容差、dedup/rim/split 的比例、灰度阈值。给它们乘上 scale 反而会误伤。
    """
    p = dict(params)
    if scale >= 1.0:
        return p
    s2 = scale * scale
    for key, (off_ok, lo) in _LENGTH_PARAMS.items():
        v = p.get(key)
        if v is None or (off_ok and float(v) <= 0):
            continue  # 0 = 关闭，折算后仍然是关闭
        p[key] = max(lo, int(round(float(v) * scale)))
    for key in _AREA_PARAMS:
        v = p.get(key)
        if v is not None:
            p[key] = float(v) * s2
    if p.get("min_r") is not None and p.get("max_r") is not None and p["max_r"] <= p["min_r"]:
        p["max_r"] = p["min_r"] + 1  # 区间塌缩时保住 Hough 的 minRadius<maxRadius
    return p

# 面板分组顺序 = 检测执行顺序：皿ROI → 预处理 → 算法(+算法参数) → 拆分 → 颜色过滤 → 过滤 → 手动
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
    ParamSpec("split_on", "拆分粘连菌落", "bool", False, group="粘连拆分",
              desc="对面积远超典型菌落的大团做距离变换+分水岭拆分（阈值/自适应/HSV 效果好；"
                   "Blob/Hough 本身按圆找，收益小）。拆出的碎片仍受面积/圆度过滤。"),
    ParamSpec("split_factor", "合并判定倍数", "float", 1.6, group="粘连拆分", min=1.2, max=4.0, step=0.1,
              desc="候选面积 > 全体中位面积×此倍数才尝试拆分。粘连漏拆→调小；"
                   "菌落大小本来就不均导致误拆→调大。"),
    ParamSpec("split_peak", "找核强度比", "float", 0.55, group="粘连拆分", min=0.3, max=0.85, step=0.05,
              desc="距离变换的局部极大处就是一个“菌落核”，核的强度低于最强峰的此比例时忽略。"
                   "该拆的没拆开→调小；单个菌落被拆两半→调大。"),
    ParamSpec("color_on", "启用颜色过滤", "bool", False, group="颜色过滤",
              desc="按取色器采到的颜色筛掉候选：与所有“菌落颜色”相差太远的去掉，"
                   "与“排除颜色”相近的去掉。拆分后、几何过滤前执行，对所有算法生效。"),
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
    ParamSpec("min_area", "最小面积(px²)", "int", 20, group="后处理过滤", min=1, max=5000, step=1,
              desc="面积小于此值的候选丢弃（含拆分出的碎片）。调小可检出细小菌落，但噪点变多。"),
    ParamSpec("max_area", "最大面积(px²)", "int", 800, group="后处理过滤", min=10, max=50000, step=10,
              desc="面积大于此值的候选丢弃。调小可排除马克笔字；拆分在过滤前执行，大团先拆再判。"),
    ParamSpec("min_circ", "最小圆度", "float", 0.35, group="后处理过滤", min=0.0, max=1.0, step=0.05,
              desc="圆度 4πA/P²，1=正圆。调大排除马克笔细长笔画；过小会把拉丝菌落当笔画滤掉。"),
    ParamSpec("max_aspect", "最大长宽比", "float", 4.0, group="后处理过滤", min=1.0, max=20.0, step=0.5,
              desc="最小外接矩形的长宽比上限，过滤笔画状/拉丝状目标。"),
    ParamSpec("dedup_on", "合并重叠菌落", "bool", True, group="后处理过滤",
              desc="两圆交集超过小圆的阈值比例（含大圆套小圆）判为同一菌落，保留大的。"
                   "Hough 出现双圈/重复圈时靠它收拾；相邻真菌落只在边缘相交，不会被误并。"),
    ParamSpec("dedup_ovl", "重叠判定阈值", "float", 0.6, group="后处理过滤", min=0.3, max=0.95, step=0.05,
              desc="交集面积/小圆面积。调小更激进（可能误并相邻菌落）；调大只清理几乎完全重叠。"
                   "包含情形重叠比=1，任何阈值都能抓到。"),
    ParamSpec("manual_r", "手动菌落半径(px)", "float", 0.0, group="手动校正", min=0.0, max=100.0, step=0.5,
              desc="“添加”模式点击空白处时新菌落的半径。0=自动跟随当前检测出的菌落半径"
                   "中位数（没有检出菌落时按 6px），想让手加的圈和周围大小一致就保持 0；"
                   "要固定大小才填具体值。"),
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
    amax_aspect = float(params.get("max_aspect", 4.0))  # 尺度无关量，不做折算（见 _to_working_params）
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
    dish_circle: Any = UNSET,
) -> tuple[list[Colony], PrepContext, float]:
    """在原图上跑检测；内部缩放到工作分辨率。返回菌落为**原图坐标**。

    返回 (colonies, ctx, scale)：scale = 工作/原图 比例，UI 用工作坐标时乘 scale。
    `ctx.dish_circle` 是本次用到的皿圆（工作坐标），可按图缓存后经 `dish_circle=` 传回，
    省下无 alpha 图上 ~173ms 的 Hough 找圆。
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, work_max / max(h, w))
    if scale < 1.0:
        wb = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        wa = cv2.resize(alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if alpha is not None else None
    else:
        wb, wa = bgr, alpha

    # 面板参数是原图 px 口径 → 折算到工作尺度，长度 ×scale、面积 ×scale²
    wp = _to_working_params(params, scale)

    if isinstance(dish_circle, _Unset):
        dish_circle = find_dish_circle(wb, wa)  # 只依赖图像，与 rim_pct 无关 → 可按图缓存
    dish_mask = dish_mask_from_circle(dish_circle, wb.shape, float(params.get("rim_pct", 0.12)))
    gray = prepare(wb, wa, dish_mask, wp)
    ctx = PrepContext(bgr=wb, gray=gray, dish_mask=dish_mask, alpha=wa, dish_circle=dish_circle)

    det = get_detector(params.get("algorithm", "adaptive"))
    raw = det.detect(ctx, wp)

    raw = split_merged(ctx, raw, wp)           # ① 粘连拆分（split_on 关闭时原样返回；也要用折算后的 block/面积）
    raw = apply_color_filter(ctx, raw, params)  # ② 颜色过滤（只看 Lab 距离，与尺度无关）
    kept = _post_filter(raw, wp, dish_mask)
    if params.get("dedup_on", True):  # ③ 重叠/包含去重（保留大圆）
        kept = dedup_colonies(kept, float(params.get("dedup_ovl", 0.6)))

    inv = 1.0 / scale
    for c in kept:
        c.x *= inv
        c.y *= inv
        c.r *= inv
        c.area *= inv * inv
    return kept, ctx, scale
