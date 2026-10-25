"""Unicode/中文路径读写回归（Windows 下 cv2.imread 按 ANSI 代码页解析中文路径失败的坑）。

注：Linux 上旧写法也能过——这些测试锁定的是新 API 行为；Windows 效果需实机验证。
"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.export import export_png
from bac_count.core.imio import imread_any, imwrite_any
from bac_count.core.models import Colony
from bac_count.core.preprocess import load_image


def _plate() -> np.ndarray:
    img = np.zeros((120, 120, 3), np.uint8)
    img[:] = (200, 220, 240)
    cv2.circle(img, (60, 60), 30, (90, 110, 137), -1)
    return img


def test_roundtrip_chinese_path(tmp_path):
    d = tmp_path / "课题_鸭子_鸭肉抑菌实验" / "平板照片_抠图"
    d.mkdir(parents=True)
    f = d / "E0-1-10-1.png"
    img = _plate()
    imwrite_any(f, img)
    assert f.exists()
    back = imread_any(f)
    assert back is not None and np.array_equal(back, img)


def test_load_image_rgba_chinese_path(tmp_path):
    d = tmp_path / "中文目录"
    d.mkdir()
    rgba = np.zeros((50, 50, 4), np.uint8)
    rgba[..., :3] = (10, 20, 30)
    rgba[10:40, 10:40, 3] = 255
    f = d / "皿图.png"
    imwrite_any(f, rgba)
    bgr, alpha = load_image(f)
    assert bgr.shape == (50, 50, 3)
    assert alpha is not None and alpha[25, 25] == 255 and alpha[0, 0] == 0


def test_export_png_chinese_path(tmp_path):
    d = tmp_path / "导出_中文"
    d.mkdir()
    out = d / "标注_图.png"
    export_png(out, _plate(), [Colony(x=60, y=60, r=30, area=2800)])
    assert imread_any(out) is not None


def test_imread_missing_returns_none(tmp_path):
    assert imread_any(tmp_path / "不存在.png") is None
