"""图像加载、培养皿 ROI、预处理链。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bac_count.core.imio import imread_any
from bac_count.core.registry import PrepContext

__all__ = ["load_image", "make_dish_mask", "prepare"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


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


def make_dish_mask(bgr: np.ndarray, alpha: np.ndarray | None, rim_pct: float) -> np.ndarray | None:
    """皿内可检测区域掩膜：优先 alpha 通道，其次 Hough 大圆，失败返回 None（全图）。

    rim_pct: 皿缘内缩比例（0-0.3），alpha 区域包含塑料皿缘，需要向内收缩。
    """
    h, w = bgr.shape[:2]
    circle = None
    if alpha is not None:
        cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            circle = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    if circle is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(h, w),
            param1=100, param2=60, minRadius=int(min(h, w) * 0.3), maxRadius=int(min(h, w) * 0.62),
        )
        if circles is not None:
            cx, cy, cr = max(circles[0], key=lambda c: c[2])
            circle = ((cx, cy), cr)
    if circle is None:
        return None
    (cx, cy), cr = circle
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(cr * (1.0 - rim_pct)), 255, -1)
    return mask


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
