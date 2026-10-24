"""核心数据模型：菌落、编辑状态、参数规格。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Colony", "ExclRegion", "EditState", "DetectionResult", "ParamSpec", "merge"]


@dataclass
class Colony:
    """一个菌落，坐标/半径/面积均为**原图坐标系**。"""

    id: int = 0
    x: float = 0.0
    y: float = 0.0
    r: float = 5.0
    area: float = 0.0
    circularity: float = 1.0
    aspect: float = 1.0
    source: Literal["auto", "manual"] = "auto"
    label: str | None = None

    def dist_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class ExclRegion:
    """手动删除产生的几何排除区（原图坐标）。"""

    x: float
    y: float
    r: float


@dataclass
class EditState:
    additions: list[Colony] = field(default_factory=list)
    exclusions: list[ExclRegion] = field(default_factory=list)

    def copy(self) -> "EditState":
        return EditState(
            additions=[Colony(**c.__dict__) for c in self.additions],
            exclusions=[ExclRegion(**e.__dict__) for e in self.exclusions],
        )


@dataclass
class DetectionResult:
    colonies: list[Colony]
    auto_total: int
    kept: int
    removed: int
    added: int

    @property
    def count(self) -> int:
        return len(self.colonies)


def _excluded(c: Colony, regions: list[ExclRegion]) -> bool:
    return any(c.dist_to(e.x, e.y) <= c.r + 0.5 * e.r for e in regions)


def merge(auto: list[Colony], edits: EditState) -> DetectionResult:
    """最终结果 = (自动检测 − 排除区) ∪ 手动添加。"""
    if not edits.exclusions:
        kept = list(auto)
    else:
        kept = [c for c in auto if not _excluded(c, edits.exclusions)]
    final = kept + list(edits.additions)
    for i, c in enumerate(final):
        c.id = i + 1
    return DetectionResult(
        colonies=final,
        auto_total=len(auto),
        kept=len(kept),
        removed=len(auto) - len(kept),
        added=len(edits.additions),
    )


@dataclass
class ParamSpec:
    """参数描述，驱动右侧面板自动生成控件。"""

    key: str
    label: str
    kind: Literal["int", "float", "combo", "bool"]
    default: Any
    group: str = "参数"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
