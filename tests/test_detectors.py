"""算法在合成平板图上的召回与抗干扰能力。"""
from __future__ import annotations

import numpy as np
import pytest

from bac_count.core.pipeline import default_params, run_detection
from bac_count.core.preprocess import load_image
from bac_count.core.registry import DETECTORS


def _recall(cols, truth, tol=1.5):
    hit = 0
    for tx, ty, tr in truth:
        if any(np.hypot(c.x - tx, c.y - ty) <= max(tr * tol, 12) for c in cols):
            hit += 1
    return hit / len(truth)


@pytest.mark.parametrize("algo", ["threshold", "adaptive", "blob"])
def test_synthetic_recall(plate, algo):
    bgr, alpha, truth = plate
    cols, _ctx, _s = run_detection(bgr, alpha, default_params(algo))
    assert _recall(cols, truth) >= 0.8, f"{algo} 召回过低: {len(cols)}/{len(truth)}"
    # 马克笔大笔画/大圈不应造成大量假阳性
    assert len(cols) <= len(truth) * 1.4


def test_hough_and_hsv_run(plate):
    bgr, alpha, truth = plate
    for algo in ("hough", "hsv"):
        cols, _ctx, _s = run_detection(bgr, alpha, default_params(algo))
        assert isinstance(cols, list)


def test_registry_has_five():
    assert {"threshold", "adaptive", "blob", "hough", "hsv"} <= set(DETECTORS)


@pytest.mark.parametrize("algo", sorted(DETECTORS))
def test_all_algos_run_on_samples(algo):
    """真实样例：所有算法不抛异常。"""
    import glob
    from pathlib import Path

    samples = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "sample" / "*.png")))
    if not samples:
        pytest.skip("无样例图")
    for f in samples:
        bgr, alpha = load_image(f)
        cols, _ctx, _s = run_detection(bgr, alpha, default_params(algo))
        assert isinstance(cols, list)


def test_default_algo_regression_on_samples():
    """默认算法(adaptive)在样例上的宽松回归基准（人工计数真值 ±40%）。"""
    from pathlib import Path

    # 真值：E0-3≈50, E12≈7, P24-2≈110, P24-3≈110, P72≈25
    expected = {"E0-3": (30, 70), "E12": (0, 12), "P24-2": (66, 154), "P24-3": (66, 154), "P72": (15, 35)}
    for key in expected:
        f = next(iter(Path(__file__).resolve().parents[1].glob(f"sample/{key}*.png")))
        bgr, alpha = load_image(f)
        cols, _ctx, _s = run_detection(bgr, alpha, default_params())
        lo, hi = expected[key]
        assert lo <= len(cols) <= hi, f"{f.name}: {len(cols)} 不在 [{lo},{hi}]"
