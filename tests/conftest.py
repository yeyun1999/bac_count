"""合成平板图 fixture：均匀背景 + 暗色圆菌落 + 马克笔样细长干扰。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_plate(seed: int = 42, size: int = 600, n_colonies: int = 40):
    rng = np.random.default_rng(seed)
    bg = np.full((size, size, 3), (200, 215, 235), np.uint8)  # BGR 奶黄色
    truth: list[tuple[int, int, int]] = []
    for _ in range(n_colonies):
        r = int(rng.integers(4, 10))
        x = int(rng.integers(r + 40, size - r - 40))
        y = int(rng.integers(r + 40, size - r - 40))
        col = tuple(int(v) for v in rng.integers(60, 110, 3))  # 暗褐色
        cv2.circle(bg, (x, y), r, col, -1, cv2.LINE_AA)
        truth.append((x, y, r))
    # 马克笔样干扰：细长笔画 + 大圈
    cv2.line(bg, (150, 180), (420, 200), (30, 30, 30), 9, cv2.LINE_AA)
    cv2.line(bg, (200, 320), (210, 470), (30, 30, 30), 8, cv2.LINE_AA)
    cv2.circle(bg, (330, 400), 45, (35, 35, 35), 7, cv2.LINE_AA)
    # 轻微噪声
    noise = rng.normal(0, 3, bg.shape[:2]).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)
    # alpha：皿内全不透明，皿缘外透明
    alpha = np.zeros((size, size), np.uint8)
    cv2.circle(alpha, (size // 2, size // 2), int(size * 0.46), 255, -1)
    return bg, alpha, truth


@pytest.fixture
def plate():
    return make_plate()
