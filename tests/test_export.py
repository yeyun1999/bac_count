"""导出边界：算法产出的 Colony 必须是可 JSON 序列化的 Python 标量。

回归用例：cv2.HoughCircles 返回 np.float32，直接塞进 Colony 后
round() 仍是 np.float32 → json.dumps 抛 TypeError，GUI 导出无反应。
类型在 Colony.__post_init__ 统一收敛。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from bac_count.core.export import colony_dict, write_full_json
from bac_count.core.models import Colony
from bac_count.core.pipeline import default_params, run_detection
from bac_count.core.registry import DETECTORS

FIELDS = ("x", "y", "r", "area", "circularity", "aspect")


def test_colony_coerces_numpy_scalars():
    c = Colony(id=np.int32(7), x=np.float32(1.234), y=np.float64(2), r=np.int16(3),
               area=np.float32(4), circularity=np.float32(0.9), aspect=np.float32(1))
    assert type(c.id) is int
    for k in FIELDS:
        assert type(getattr(c, k)) is float, f"{k} 未被收敛为 Python float"
    assert json.dumps(colony_dict(c))  # 修复前这里 TypeError


@pytest.mark.parametrize("algo", sorted(DETECTORS))
def test_all_algos_produce_json_serializable_colonies(plate, algo):
    bgr, alpha, _ = plate
    cols, _ctx, _s = run_detection(bgr, alpha, default_params(algo))
    for c in cols:
        for k in FIELDS:
            assert type(getattr(c, k)) is float, f"{algo}: {k} 是 {type(getattr(c, k)).__name__}"
    json.dumps([colony_dict(c) for c in cols])


def test_write_full_json_hough_roundtrip(tmp_path, plate):
    bgr, alpha, _ = plate
    cols, _ctx, _s = run_detection(bgr, alpha, default_params("hough"))
    out = tmp_path / "out.json"
    write_full_json(out, [{
        "path": str(out), "name": "plate.png", "count": len(cols),
        "params": default_params("hough"),
        "colonies": [colony_dict(c) for c in cols],
        "edits": {"additions": [], "exclusions": []},
    }])
    assert json.loads(out.read_text(encoding="utf-8"))["total_count"] == len(cols)
