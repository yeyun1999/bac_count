"""图像加载、培养皿 ROI、预处理链。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bac_count.core.imio import imread_any
from bac_count.core.registry import PrepContext

__all__ = ["load_image", "find_dish_circle", "dish_mask_from_circle", "make_dish_mask", "prepare"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# 找皿（无 alpha 时的 Hough 兜底）参数。取值依据：皿基本占满画面，所以半径限定在
# [0.30, 0.62]×短边；minDist=短边 → 整图最多留一个候选，再按半径取最大者。
# param2=60 比检测菌落的 Hough（hough.py 里可调的 16）保守得多——皿缘那圈高对比
# 边缘很容易投票，阈值低了会贴到培养皿之外的桌面边框上。
_DISH_DP = 1.5
_DISH_PARAM1 = 100
_DISH_PARAM2 = 60
_DISH_R_LO = 0.30  # × min(h, w)
_DISH_R_HI = 0.62  # × min(h, w)
_DISH_MEDIAN_K = 7  # Hough 前的降噪核（奇数）


def load_image(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """读取为 BGR + alpha（0/255 uint8 或 None）。中文路径安全（core/imio）。"""
    raw = imread_any(path)
    if raw is None:
        raise ValueError(f"无法读取图片: {path}")
    if raw.dtype == np.uint16:
        raw = (raw // 256).astype(np.uint8)
    alpha = None
    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr = raw[:, :, :3].copy()
        alpha = np.where(raw[:, :, 3] > 10, 255, 0).astype(np.uint8)
    elif raw.ndim == 3:
        bgr = raw
    else:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return bgr, alpha


def find_dish_circle(bgr: np.ndarray, alpha: np.ndarray | None) -> tuple[tuple[float, float], float] | None:
    """皿圆（**输入图坐标**）：优先 alpha 外轮廓的最小外接圆，其次 Hough 大圆，找不到返回 None。

    只依赖图像内容，与 `rim_pct` 无关 → 同一张图可以只算一次缓存起来
    （无 alpha 的手机照片这一步是 Hough，1200px 实测 ~173ms，占整条流水线 71%）。
    """
    h, w = bgr.shape[:2]
    if alpha is not None:
        cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (cx, cy), cr = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
            return ((float(cx), float(cy)), float(cr))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, _DISH_MEDIAN_K)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=_DISH_DP, minDist=min(h, w),
        param1=_DISH_PARAM1, param2=_DISH_PARAM2,
        minRadius=int(min(h, w) * _DISH_R_LO), maxRadius=int(min(h, w) * _DISH_R_HI),
    )
    if circles is None:
        return None
    cx, cy, cr = max(circles[0], key=lambda c: c[2])
    return ((float(cx), float(cy)), float(cr))


def dish_mask_from_circle(circle: tuple[tuple[float, float], float] | None,
                          shape: tuple[int, ...], rim_pct: float) -> np.ndarray | None:
    """纯几何：皿圆 + 内缩比例 → 皿内可检测区域掩膜（0/255），circle 为 None 时返回 None（全图）。

    rim_pct: 皿缘内缩比例（0-0.3），alpha 区域包含塑料皿缘，需要向内收缩。
    """
    if circle is None:
        return None
    h, w = shape[:2]
    (cx, cy), cr = circle
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(cr * (1.0 - rim_pct)), 255, -1)
    return mask


def make_dish_mask(bgr: np.ndarray, alpha: np.ndarray | None, rim_pct: float) -> np.ndarray | None:
    """皿内可检测区域掩膜：找圆 + 内缩一步到位。

    兼容入口；调参热路径请拆成 `find_dish_circle`（可按图缓存）+ `dish_mask_from_circle`。
    """
    return dish_mask_from_circle(find_dish_circle(bgr, alpha), bgr.shape, rim_pct)


def prepare(bgr: np.ndarray, alpha: np.ndarray | None, dish_mask: np.ndarray | None, pp: dict) -> np.ndarray:
    """通道选择 → 阴影校正 → 高斯模糊 → CLAHE → 极性统一（菌落恒为暗色）。"""
    ch = pp.get("channel", "gray")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if ch == "gray":
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif ch == "red":
        img = bgr[:, :, 2]
    elif ch == "green":
        img = bgr[:, :, 1]
    elif ch == "blue":
        img = bgr[:, :, 0]
    elif ch == "hsv_h":
        img = hsv[:, :, 0]
    elif ch == "hsv_s":
        img = hsv[:, :, 1]
    elif ch == "hsv_v":
        img = hsv[:, :, 2]
    else:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if dish_mask is not None:
        # 皿外（RGBA 透明区 RGB 常为纯黑）用皿内中值填充，避免污染 Otsu/统计
        inner = dish_mask > 0
        if inner.any():
            img = np.where(inner, img, int(np.median(img[inner]))).astype(np.uint8)

    shading = int(pp.get("shading", 0))
    if shading >= 3:
        k = shading | 1
        bg = cv2.medianBlur(img, min(k, 151))
        img = np.clip(img.astype(np.int16) - bg.astype(np.int16) + 128, 0, 255).astype(np.uint8)

    blur = int(pp.get("blur_k", 3))
    if blur >= 1:
        img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)

    if pp.get("clahe", False):
        clahe = cv2.createCLAHE(clipLimit=float(pp.get("clahe_clip", 2.0)), tileGridSize=(8, 8))
        img = clahe.apply(img)

    if pp.get("polarity", "dark") == "light":
        img = 255 - img
    return img
