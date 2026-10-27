"""核心数据模型：菌落、编辑状态、参数规格。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Colony", "ExclRegion", "EditState", "DetectionResult", "ParamSpec", "merge"]

_EXCL_MIN_R = 3.0  # 排除区的最小命中半径：参数改动后重检，同一菌落中心会漂几像素


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

    def __post_init__(self) -> None:
        # 数据边界收敛类型：cv2.HoughCircles / KeyPoint 等返回 np.float32，
        # 原样存下会让 round() 返回 np.float32 → json.dumps 抛 TypeError，
        # 且下游算术（dedup、draw_overlay）也带上 numpy 标量语义。
        self.id = int(self.id)
        self.x = float(self.x)
        self.y = float(self.y)
        self.r = float(self.r)
        self.area = float(self.area)
        self.circularity = float(self.circularity)
        self.aspect = float(self.aspect)

    def dist_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class ExclRegion:
    """手动删除产生的几何排除区（原图坐标）。

    `r` 存**被点掉那个菌落自身的半径**（可直接拿去画圈可视化），
    不含点击容差——容差只用于命中判定，见 `edits.EditHistory._hit`。
    """

    x: float
    y: float
    r: float

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.r = float(self.r)


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
    """候选是否**就是**被点掉的那个菌落。

    判据只用排除区自身半径（**不加 `c.r`**）：旧写法 `dist <= c.r + 0.5 * e.r` 里
    r=6 的菌落会被 e.r=10 的排除区在 11px 外抓到，而菌落直径才 ~12px ——
    密集平板上"删一个"会连带吃掉紧邻的真菌落，且排除区在画布上不可见，用户无从察觉。
    下限 `_EXCL_MIN_R` 留给"重新检测后同一菌落中心略偏移"的情况（见 test_exclusion_survives_redetect）。
    """
    return any(c.dist_to(e.x, e.y) <= max(e.r, _EXCL_MIN_R) for e in regions)


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
    desc: str = ""  # 悬浮解释（tooltip）
