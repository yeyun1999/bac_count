"""粘连拆分（距离变换 + 分水岭）测试。"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.pipeline import default_params, run_detection


def _params(**kw):
    p = default_params("adaptive")
    p["split_on"] = True
    p.update(kw)
    return p


def _scene(centers, r=10, bg=200, fg=120, size=300):
    img = np.full((size, size), bg, np.uint8)
    for x, y in centers:
        cv2.circle(img, (x, y), r, fg, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _new_subsets(off, on):
    """拆分结果里相对原候选新增的子菌落。"""
    return [c for c in on if not any(
        (c.x - k.x) ** 2 + (c.y - k.y) ** 2 < 1.0 and abs(c.area - k.area) < 1 for k in off)]


def test_split_off_by_default():
    assert default_params("adaptive")["split_on"] is False


def test_split_merged_pair():
    """3 单菌落 + 1 对粘连（中心距 1.3R）→ 拆分后应为 3 单 + 2 半。"""
    bgr = _scene([(80, 150), (220, 150), (150, 60), (150, 225), (163, 225)])
    off = run_detection(bgr, None, default_params("adaptive"))[0]
    on = run_detection(bgr, None, _params())[0]
    assert len(off) == 4, f"基线应为 3 单 + 1 粘连团，实得 {len(off)}"
    assert len(on) == 5, f"拆分后应为 3 单 + 2 半，实得 {len(on)}"
    assert len([c for c in on if 140 < c.x < 175 and 210 < c.y < 240]) == 2


def test_split_triple_chain():
    """三个一串（中心距 1.3R）：拆分数由核数决定，可以 2 也可以 3，但不得丢菌落。"""
    bgr = _scene([(80, 80), (240, 240), (140, 250), (153, 250), (166, 250)])
    off = run_detection(bgr, None, default_params("adaptive"))[0]
    on = run_detection(bgr, None, _params())[0]
    assert len(off) == 3, f"基线应为 2 单 + 1 三连团，实得 {len(off)}"
    assert len(on) >= 4, f"三连团没拆出多块：{len(off)}→{len(on)}"


def test_split_no_false_split_on_singles():
    """全单菌落场景不得过拆。"""
    bgr = _scene([(80, 80), (200, 80), (140, 150), (80, 220), (220, 220)])
    on = run_detection(bgr, None, _params())[0]
    assert len(on) == 5, f"单菌落被过拆：{len(on)}"


def test_split_large_single_not_split():
    """单个但偏大的菌落（面积已过阈值）只有单峰，不得被拆。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y, r in [(60, 80, 7), (150, 80, 7), (240, 80, 7), (150, 210, 14)]:
        cv2.circle(img, (x, y), r, 120, -1)
    on = run_detection(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None,
                       _params(split_factor=1.3))[0]
    assert len(on) == 4, f"大单菌落被误拆：{[(round(c.x), round(c.y)) for c in on]}"


def test_split_peak_ratio_drops_weak_core():
    """大小对（12+9，dt 峰 11.9 vs 8.9）：split_peak 0.5 两个核都算→拆 2；
    0.8 时小核被判弱峰→整体作废、保留原团。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y, r in [(60, 150, 12), (240, 150, 12), (150, 225, 12), (164, 225, 9)]:
        cv2.circle(img, (x, y), r, 120, -1)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    kw = dict(split_factor=1.15)
    loose = run_detection(bgr, None, _params(split_peak=0.5, **kw))[0]
    strict = run_detection(bgr, None, _params(split_peak=0.8, **kw))[0]
    assert len(loose) == 4, f"大小对没拆成两块：{[(round(c.x), round(c.y), round(c.r)) for c in loose]}"
    assert len(strict) == 3, f"弱核应被忽略、保留原团：{[(round(c.x), round(c.y), round(c.r)) for c in strict]}"


def test_split_on_samples_no_explosion():
    """真实样例：开拆分不崩溃、计数不失控（≤2.5×关拆分）。"""
    import glob
    from pathlib import Path

    from bac_count.core.preprocess import load_image

    for f in sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "sample" / "*.png"))):
        bgr, alpha = load_image(f)
        off = len(run_detection(bgr, alpha, default_params("adaptive"))[0])
        on = len(run_detection(bgr, alpha, _params())[0])
        assert on <= max(off * 2.5, off + 10), f"{Path(f).name}: {off} → {on} 异常膨胀"
