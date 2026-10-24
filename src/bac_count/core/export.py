"""导出：CSV / 标注 PNG / JSON 状态。"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from bac_count.core.models import Colony, EditState

__all__ = ["draw_overlay", "export_png", "export_csv", "export_json"]

COLORS = {"auto": (0, 230, 0), "manual": (0, 160, 255)}  # BGR: 绿=自动 橙=手动


def draw_overlay(bgr: np.ndarray, colonies: list[Colony], scale: float = 1.0, show_ids: bool = True) -> np.ndarray:
    img = bgr.copy()
    for c in colonies:
        col = COLORS.get(c.source, (0, 230, 0))
        x, y, r = int(c.x * scale), int(c.y * scale), max(2, int(c.r * scale))
        cv2.circle(img, (x, y), r + 2, col, 1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(img, str(c.id), (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * max(scale, 0.5), col, 1, cv2.LINE_AA)
    return img


def export_png(path: str | Path, bgr: np.ndarray, colonies: list[Colony], scale: float = 1.0) -> None:
    cv2.imwrite(str(path), draw_overlay(bgr, colonies, scale))


def export_csv(path: str | Path, colonies: list[Colony], image_name: str = "") -> None:
    p = Path(path)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "id", "x", "y", "radius", "area", "source", "label"])
        for c in colonies:
            w.writerow([image_name, c.id, f"{c.x:.1f}", f"{c.y:.1f}", f"{c.r:.1f}", f"{c.area:.1f}", c.source, c.label or ""])


def export_json(path: str | Path, colonies: list[Colony], edits: EditState, params: dict, image_name: str = "") -> None:
    p = Path(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    data = {
        "image": image_name,
        "count": len(colonies),
        "params": params,
        "colonies": [asdict(c) for c in colonies],
        "edits": {
            "additions": [asdict(c) for c in edits.additions],
            "exclusions": [asdict(e) for e in edits.exclusions],
        },
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
