"""手动增删 / 排除区持久化 / 撤销重做。"""
from __future__ import annotations

from bac_count.core.edits import EditHistory
from bac_count.core.models import Colony, merge


def _auto(n=5):
    return [Colony(id=i + 1, x=100 + i * 50, y=100, r=8, area=200) for i in range(n)]


def test_add_kept_across_redetect():
    h = EditHistory()
    h.add_colony(500, 500, 6)
    r1 = merge(_auto(3), h.state)
    r2 = merge(_auto(10), h.state)  # 重新检测
    manual = [c for c in r2.colonies if c.source == "manual"]
    assert len(manual) == 1
    assert r1.count + 7 == r2.count


def test_exclusion_survives_redetect():
    h = EditHistory()
    auto = _auto(5)
    r = merge(auto, h.state)
    assert h.remove_at(auto[2].x, auto[2].y, r.colonies)
    # 再次检测：位置略偏移的同款菌落仍应被排除
    shifted = [Colony(id=i + 1, x=100 + i * 50 + 2, y=101, r=8, area=200) for i in range(5)]
    r2 = merge(shifted, h.state)
    assert r2.count == 4
    assert r2.removed == 1


def test_remove_manual_colony():
    h = EditHistory()
    h.add_colony(30, 40, 5)
    r = merge([], h.state)
    assert h.remove_at(30, 40, r.colonies)
    r2 = merge([], h.state)
    assert r2.count == 0
    assert r2.added == 0


def test_undo_redo():
    h = EditHistory()
    h.add_colony(1, 1, 5)
    h.add_colony(2, 2, 5)
    assert merge(_auto(3), h.state).count == 5
    h.undo()
    assert merge(_auto(3), h.state).count == 4
    h.redo()
    assert merge(_auto(3), h.state).count == 5


def test_ids_sequential():
    h = EditHistory()
    h.add_colony(999, 999, 5)
    r = merge(_auto(3), h.state)
    assert [c.id for c in r.colonies] == [1, 2, 3, 4]
