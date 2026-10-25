"""导出：标注 PNG / 简要汇总 CSV（一图一行）/ 完整 JSON（路径+参数+菌落）。"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from bac_count.core.imio import imwrite_any
from bac_count.core.models import Colony, EditState

__all__ = ["draw_overlay", "export_png", "write_summary_csv", "write_full_json"]

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
    """单张标注图（批量导出时由调用方逐张调用）。中文路径安全（core/imio）。"""
    imwrite_any(path, draw_overlay(bgr, colonies, scale))


def write_summary_csv(path: str | Path, rows: list[dict]) -> None:
    """rows: [{name, count, auto_total, added, removed}, ...] 一图一行。"""
    p = Path(path)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "菌落数量", "自动检出", "手动添加", "手动排除"])
        for r in rows:
            w.writerow([r["name"], r["count"], r["auto_total"], r["added"], r["removed"]])


def write_full_json(path: str | Path, items: list[dict]) -> None:
    """items: [{path, name, count, params, colonies, edits}, ...] 合并为单个 JSON。"""
    p = Path(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    data = {
        "tool": "bac_count",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "image_count": len(items),
        "total_count": sum(i["count"] for i in items),
        "images": items,
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def colony_dict(c: Colony) -> dict:
    return {
        "id": c.id, "x": round(c.x, 2), "y": round(c.y, 2), "r": round(c.r, 2),
        "area": round(c.area, 1), "source": c.source, "label": c.label,
    }


def edits_dict(e: EditState) -> dict:
    return {
        "additions": [colony_dict(c) for c in e.additions],
        "exclusions": [{"x": round(x.x, 2), "y": round(x.y, 2), "r": round(x.r, 2)} for x in e.exclusions],
    }
