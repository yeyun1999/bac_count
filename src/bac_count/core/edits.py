"""手动编辑操作 + 撤销/重做（纯逻辑，无 Qt 依赖）。"""
from __future__ import annotations

from statistics import median

from bac_count.core.models import Colony, EditState, ExclRegion

__all__ = ["EditHistory", "DEFAULT_MANUAL_R", "resolve_manual_r"]

DEFAULT_MANUAL_R = 6.0  # 半径自动模式下没有任何已检出菌落时的兜底


def resolve_manual_r(colonies: list[Colony], configured: float) -> float:
    """「手动菌落半径」的实际取值。

    面板值 >0 → 就按它；**0（默认）= 跟随当前自动检出菌落半径的中位数**，
    这样点出来的手动菌落和周围的大小看着一致，换一种分辨率/另一种平板都不用重设。
    没有任何已检出菌落（空列表，或半径都被夹到 0）时退回 `DEFAULT_MANUAL_R`。
    `colonies` 传自动检出结果（不是 merge 后的列表，否则自己加的菌集会参与中位数）。
    """
    r = float(configured or 0.0)
    if r > 0:
        return r
    radii = [c.r for c in colonies if c.r > 0]
    return float(median(radii)) if radii else DEFAULT_MANUAL_R


class EditHistory:
    """每张图一个实例。所有编辑通过 add/remove 方法走，自动记录快照。"""

    def __init__(self, limit: int = 100):
        self.state = EditState()
        self._undo: list[EditState] = []
        self._redo: list[EditState] = []
        self._limit = limit

    def _snapshot(self) -> None:
        self._undo.append(self.state.copy())
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def add_colony(self, x: float, y: float, r: float) -> Colony:
        self._snapshot()
        c = Colony(x=x, y=y, r=r, area=3.14159 * r * r, source="manual")
        self.state.additions.append(c)
        return c

    def remove_at(self, x: float, y: float, colonies: list[Colony]) -> bool:
        """删除模式下点击：命中最近的菌落。手动菌落直接移除；自动菌落记排除区。

        排除区几何 = **命中菌落自身**的 (x, y, r)；点击容差（`_hit` 里的
        `max(c.r + 4, 10)`）只用来决定"点中了谁"，不参与排除区大小，
        否则删一个会连带吃掉邻居（见 models._excluded）。
        """
        hit = self._hit(x, y, colonies)
        if hit is None:
            return False
        self._snapshot()
        if hit.source == "manual" and hit in self.state.additions:
            self.state.additions.remove(hit)
        else:
            self.state.exclusions.append(ExclRegion(x=hit.x, y=hit.y, r=hit.r))
        return True

    @staticmethod
    def _hit(x: float, y: float, colonies: list[Colony]) -> Colony | None:
        best, best_d = None, None
        for c in colonies:
            d = c.dist_to(x, y)
            if d <= max(c.r + 4.0, 10.0) and (best_d is None or d < best_d):
                best, best_d = c, d
        return best

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.state.copy())
        self.state = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.state.copy())
        self.state = self._redo.pop()
        return True

    def clear_edits(self) -> None:
        self._snapshot()
        self.state = EditState()
