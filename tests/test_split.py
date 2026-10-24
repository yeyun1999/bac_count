"""粘连拆分算法测试。"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.pipeline import default_params, run_detection


def _scene() -> np.ndarray:
    """3 个单菌落 + 1 对粘连（r=10，中心距 13 → 面积≈1.77×单粒）。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y in [(80, 150), (220, 150), (150, 60)]:
        cv2.circle(img, (x, y), 10, 120, -1)
    cv2.circle(img, (150, 225), 10, 120, -1)
    cv2.circle(img, (163, 225), 10, 120, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_split_off_by_default():
    assert default_params("adaptive")["split_on"] is False


def test_split_merged_pair():
    bgr = _scene()
    off = run_detection(bgr, None, default_params("adaptive"))[0]
    p = default_params("adaptive")
    p["split_on"] = True
    on = run_detection(bgr, None, p)[0]
    assert len(off) == 4, f"基线应为 3 单 + 1 粘连团，实得 {len(off)}"
    assert len(on) == 5, f"拆分后应为 3 单 + 2 半，实得 {len(on)}"
    # 拆出的两半应落在粘连团附近
    near = [c for c in on if 140 < c.x < 175 and 210 < c.y < 240]
    assert len(near) == 2


def test_split_tight_pair():
    """半重叠对（中心距 = 半径）也能拆；更紧的粘连超出分水岭物理极限，靠手动。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y in [(80, 100), (220, 100), (100, 220)]:
        cv2.circle(img, (x, y), 10, 120, -1)
    cv2.circle(img, (150, 150), 10, 120, -1)
    cv2.circle(img, (160, 150), 10, 120, -1)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = default_params("adaptive")
    p["split_on"] = True
    on = run_detection(bgr, None, p)[0]
    assert len(on) == 5, f"半重叠对应拆为 2，实得总数 {len(on)}"


def test_split_no_false_split_on_singles():
    """全单菌落场景不得过拆。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y in [(80, 80), (200, 80), (140, 150), (80, 220), (220, 220)]:
        cv2.circle(img, (x, y), 10, 120, -1)
    p = default_params("adaptive")
    p["split_on"] = True
    on = run_detection(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, p)[0]
    assert len(on) == 5, f"单菌落被过拆：{len(on)}"


def test_split_on_samples_no_explosion():
    """真实样例：开拆分不崩溃、计数不失控（≤2.5×关拆分）。"""
    import glob
    from pathlib import Path

    samples = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "sample" / "*.png")))
    from bac_count.core.preprocess import load_image

    for f in samples:
        bgr, alpha = load_image(f)
        off = len(run_detection(bgr, alpha, default_params("adaptive"))[0])
        p = default_params("adaptive")
        p["split_on"] = True
        on = len(run_detection(bgr, alpha, p)[0])
        assert on <= max(off * 2.5, off + 10), f"{Path(f).name}: {off} → {on} 异常膨胀"
