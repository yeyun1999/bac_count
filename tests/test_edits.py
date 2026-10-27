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


# ------------------------------------------------------------------ §1.5 排除区判据
def test_deleting_one_colony_spares_its_neighbours():
    """密集平板上删中间那个，只该少一个。

    回归：`_excluded` 旧判据 `dist <= c.r + 0.5 * e.r`，而排除区半径写的是
    `max(hit.r, 10)` → r=6 的菌落连 11px 内的邻居一起吃掉（菌落直径才 ~12px），
    且排除区在画布上不可见，用户根本不知道自己删多了。
    """
    auto = [Colony(id=i + 1, x=100 + i * 10, y=100, r=6, area=113) for i in range(3)]
    h = EditHistory()
    r = merge(auto, h.state)
    assert h.remove_at(auto[1].x, auto[1].y, r.colonies), "点不中中间那个"
    r2 = merge(auto, h.state)
    assert r2.removed == 1 and r2.count == 2
    assert [(c.x, c.y) for c in r2.colonies] == [(100.0, 100.0), (120.0, 100.0)]


def test_bigger_colonies_touching_at_8px_apart_survive():
    """中心距 8px、半径 6 的两个菌落：删掉一个，另一个必须活着（旧判据阈值 9~11px）。"""
    auto = [Colony(id=1, x=100, y=100, r=6, area=113), Colony(id=2, x=108, y=100, r=6, area=113)]
    h = EditHistory()
    h.remove_at(100, 100, merge(auto, h.state).colonies)
    r = merge(auto, h.state)
    assert r.count == 1 and r.colonies[0].x == 108


def test_exclusion_region_stores_hit_geometry_only():
    """排除区几何 = 命中菌落自身的 (x, y, r)，不掺点击容差/manual_r。

    这个几何以后要直接拿去画半透明红圈（§6.1 排除区可视化），
    掺了容差就会画出比实际影响范围大的圈。
    """
    from bac_count.core.models import ExclRegion

    auto = [Colony(id=1, x=100, y=100, r=4, area=50)]
    h = EditHistory()
    assert h.remove_at(103, 100, merge(auto, h.state).colonies)  # 靠点击容差命中
    assert h.state.exclusions == [ExclRegion(x=100.0, y=100.0, r=4.0)]


def test_small_exclusion_does_not_grab_distant_big_colony():
    """判据不加 `c.r`：大菌落不该被远处的小排除区抓到。

    旧写法阈值 = 10 + 0.5×3 = 11.5，圆心距 11 的 r=10 大菌落会被误删。
    """
    from bac_count.core.models import EditState, ExclRegion

    big = Colony(id=1, x=11, y=0, r=10, area=314)
    assert big.dist_to(0, 0) == 11
    r = merge([big], EditState(exclusions=[ExclRegion(x=0, y=0, r=3)]))
    assert r.count == 1 and r.removed == 0


# ------------------------------------------------------------------ 手动半径（0=自动中位数）
def test_resolve_manual_r_uses_median_of_auto():
    from bac_count.core.edits import resolve_manual_r

    cols = [Colony(r=4), Colony(r=6), Colony(r=8), Colony(r=10)]
    assert resolve_manual_r(cols, 0.0) == 7.0          # 偶数个取中间两项均值
    assert resolve_manual_r([Colony(r=5), Colony(r=9), Colony(r=7)], 0) == 7.0


def test_resolve_manual_r_explicit_value_wins():
    from bac_count.core.edits import resolve_manual_r

    assert resolve_manual_r([Colony(r=4), Colony(r=4)], 12.5) == 12.5
    assert resolve_manual_r([], 3.0) == 3.0  # 显式值优先，哪怕没有已检出菌落


def test_resolve_manual_r_falls_back_when_nothing_detected():
    from bac_count.core.edits import DEFAULT_MANUAL_R, resolve_manual_r

    assert DEFAULT_MANUAL_R == 6.0
    assert resolve_manual_r([], 0) == DEFAULT_MANUAL_R
    assert resolve_manual_r([], None) == DEFAULT_MANUAL_R  # 预设缺键也不炸
    assert resolve_manual_r([Colony(r=0.0)], 0) == DEFAULT_MANUAL_R  # 半径全被夹到 0


def test_manual_r_default_is_auto_mode():
    from bac_count.core.pipeline import default_params

    assert default_params()["manual_r"] == 0.0
    assert default_params("hough")["manual_r"] == 0.0
