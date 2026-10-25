"""粘连拆分算法测试。"""
from __future__ import annotations

import cv2
import numpy as np

from bac_count.core.pipeline import default_params, run_detection


def _scene() -> np.ndarray:
    """3 个单菌落 + 1 对粘连（r=10，中心距 13 → 面积≈1.77×单粒）。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y in [(80, 150), (220, 150), (150, 60)]:
        cv2.circle(img, (x, y), 10, 120, -1)
    cv2.circle(img, (150, 225), 10, 120, -1)
    cv2.circle(img, (163, 225), 10, 120, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_split_off_by_default():
    assert default_params("adaptive")["split_on"] is False


def test_split_merged_pair():
    bgr = _scene()
    off = run_detection(bgr, None, default_params("adaptive"))[0]
    p = default_params("adaptive")
    p["split_on"] = True
    on = run_detection(bgr, None, p)[0]
    assert len(off) == 4, f"基线应为 3 单 + 1 粘连团，实得 {len(off)}"
    assert len(on) == 5, f"拆分后应为 3 单 + 2 半，实得 {len(on)}"
    # 拆出的两半应落在粘连团附近
    near = [c for c in on if 140 < c.x < 175 and 210 < c.y < 240]
    assert len(near) == 2


def test_split_tight_pair():
    """半重叠对（中心距 = 半径）也能拆；更紧的粘连超出分水岭物理极限，靠手动。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y in [(80, 100), (220, 100), (100, 220)]:
        cv2.circle(img, (x, y), 10, 120, -1)
    cv2.circle(img, (150, 150), 10, 120, -1)
    cv2.circle(img, (160, 150), 10, 120, -1)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = default_params("adaptive")
    p["split_on"] = True
    on = run_detection(bgr, None, p)[0]
    assert len(on) == 5, f"半重叠对应拆为 2，实得总数 {len(on)}"


def test_split_no_false_split_on_singles():
    """全单菌落场景不得过拆。"""
    img = np.full((300, 300), 200, np.uint8)
    for x, y in [(80, 80), (200, 80), (140, 150), (80, 220), (220, 220)]:
        cv2.circle(img, (x, y), 10, 120, -1)
    p = default_params("adaptive")
    p["split_on"] = True
    on = run_detection(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, p)[0]
    assert len(on) == 5, f"单菌落被过拆：{len(on)}"


def test_split_on_samples_no_explosion():
    """真实样例：开拆分不崩溃、计数不失控（≤2.5×关拆分）。"""
    import glob
    from pathlib import Path

    samples = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "sample" / "*.png")))
    from bac_count.core.preprocess import load_image

    for f in samples:
        bgr, alpha = load_image(f)
        off = len(run_detection(bgr, alpha, default_params("adaptive"))[0])
        p = default_params("adaptive")
        p["split_on"] = True
        on = len(run_detection(bgr, alpha, p)[0])
        assert on <= max(off * 2.5, off + 10), f"{Path(f).name}: {off} → {on} 异常膨胀"


def test_split_rejects_marker_stroke_fragment():
    """回归（用户 E0-1 实测）：菌落+马克笔粗划合并轮廓，拆分不得产出落在笔画上的异色碎片。

    拆分的内部二值图是色盲的 adaptive 阈值（含笔画），无颜色守卫时笔画粗段会被
    切成"矮胖块"通过几何过滤，凭空多出一个菌落。
    """
    import cv2

    from bac_count.core.registry import PrepContext, get_detector
    from bac_count.core.split import split_merged

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (200, 200, 200)
    BROWN = (90, 115, 140)  # bgr 褐菌落色
    cv2.circle(img, (150, 100), 8, BROWN, -1)
    cv2.rectangle(img, (150, 94), (190, 107), (30, 30, 30), -1)  # 贴住菌落的黑粗划
    cv2.circle(img, (80, 180), 8, BROWN, -1)
    cv2.circle(img, (220, 180), 8, BROWN, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = PrepContext(bgr=img, gray=gray, dish_mask=None, alpha=None)

    p = default_params("adaptive")
    p["split_on"] = True
    raw = get_detector("adaptive").detect(ctx, p)
    assert len(raw) == 3, f"应有 2 单菌落 + 1 合并轮廓，实得 {len(raw)}"
    out = split_merged(ctx, raw, p)
    assert len(out) == 3, f"颜色守卫失效，产生了笔画碎片：{[(round(c.x), round(c.y), round(c.r)) for c in out]}"
    # 笔画中心附近不得出现小半径"菌落"
    assert all(((c.x - 170) ** 2 + (c.y - 100) ** 2) ** 0.5 > 15 or c.r > 15 for c in out)


def test_split_dark_overlap_pair_still_splits():
    """回归（用户 E0-1 实拍）：半透明菌落重叠区颜色更深，种子色=深色带，
    两个浅色碎片靠兄弟规则互认，不得被颜色守卫全军覆没。

    直接喂干净二值图给 _split_one，隔离 adaptive 阈值对合成场景的干扰。
    """
    import cv2

    from bac_count.core.models import Colony
    from bac_count.core.registry import PrepContext
    from bac_count.core.split import _split_one

    BROWN = (150, 120, 95)
    DARK = (70, 45, 30)
    bgr = np.zeros((300, 300, 3), np.uint8)
    bgr[:, :] = (215, 195, 170)
    cv2.circle(bgr, (138, 150), 14, BROWN, -1)
    cv2.circle(bgr, (166, 150), 14, BROWN, -1)
    cv2.rectangle(bgr, (148, 142), (156, 158), DARK, -1)  # 深色重叠透镜区
    mask = np.zeros((300, 300), np.uint8)
    cv2.circle(mask, (138, 150), 14, 255, -1)
    cv2.circle(mask, (166, 150), 14, 255, -1)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ctx = PrepContext(bgr=bgr, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                      dish_mask=None, alpha=None)
    cand = Colony(x=152, y=150, r=15, area=1500)
    p = default_params("adaptive")
    p["split_on"] = True
    subs = _split_one(ctx, mask, dt, cand, p)
    assert subs is not None and len(subs) == 2, f"兄弟颜色规则失效：{subs}"
    xs = sorted(round(c.x) for c in subs)
    assert 130 <= xs[0] <= 145 and 158 <= xs[1] <= 172


def test_split_stray_rejected_by_overlap_toggle():
    """同色邻接异物（crop 混入的独立方块）：重叠校验独立拦截；关开关则放行。"""
    import cv2

    from bac_count.core.registry import PrepContext, get_detector
    from bac_count.core.split import split_merged

    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (215, 195, 170)
    BROWN = (150, 120, 95)
    cv2.circle(img, (138, 150), 14, BROWN, -1)
    cv2.circle(img, (166, 150), 14, BROWN, -1)          # 粘连对（候选）
    cv2.rectangle(img, (143, 115), (161, 133), BROWN, -1)  # 同色独立方块，间隙 3px
    cv2.circle(img, (60, 250), 14, BROWN, -1)
    cv2.circle(img, (250, 250), 14, BROWN, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = PrepContext(bgr=img, gray=gray, dish_mask=None, alpha=None)
    p = default_params("threshold")
    p["split_on"] = True
    raw = get_detector("threshold").detect(ctx, p)
    assert len(raw) == 4, f"场景应为 2 单菌落 + 1 粘连对 + 1 方块，实得 {len(raw)}"

    def near_sq(cols):
        return [c for c in cols if 140 < c.x < 165 and 112 < c.y < 138]

    out_on = split_merged(ctx, raw, p)
    assert len(out_on) == 5, f"开校验应剔除异物碎片：{[(round(c.x), round(c.y)) for c in out_on]}"
    assert len(near_sq(out_on)) == 1, "方块只保留检测器原轮廓，不得多出拆分碎片"
    halves = [c for c in out_on if 130 < c.x < 175 and 140 < c.y < 160]
    assert len(halves) == 2, "粘连对本身应正常拆分"

    out_off = split_merged(ctx, raw, dict(p, split_overlap=False, split_color=False))
    assert len(out_off) == 6, "关闭两校验后应回到旧行为（异物碎片混入），证明开关生效"
    assert len(near_sq(out_off)) == 2, "关校验后方块处应出现重复碎片"


def test_split_guard_toggles_default_on():
    p = default_params("adaptive")
    assert p["split_overlap"] is True and p["split_color"] is True


def test_split_arms_rejected_by_shape_gate():
    """同色同连通域的长条臂碎片（笔画交叉类，用户 E0-1 位置2）：只有形状校验能拦。

    两根 14×45 棕色条经 6×4 细桥连通：峰值可分（桥 dt<阈值）、同色（颜色校验过）、
    同域（重叠校验过）——开形状校验必须拒绝（asp 3.2>2.6），关掉则复现旧假阳性。
    """
    import cv2

    from bac_count.core.models import Colony
    from bac_count.core.registry import PrepContext
    from bac_count.core.split import _split_one

    BROWN = (150, 120, 95)
    bgr = np.zeros((300, 300, 3), np.uint8)
    bgr[:, :] = (215, 195, 170)
    mask = np.zeros((300, 300), np.uint8)
    for rect in [((120, 143), (134, 188)), ((137, 143), (151, 188)), ((133, 163), (139, 167))]:
        cv2.rectangle(bgr, rect[0], rect[1], BROWN, -1)
        cv2.rectangle(mask, rect[0], rect[1], 255, -1)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ctx = PrepContext(bgr=bgr, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                      dish_mask=None, alpha=None)
    cand = Colony(x=135, y=165, r=24, area=1250)
    p = default_params("adaptive")
    p["split_on"] = True

    subs_on = _split_one(ctx, mask, dt, cand, p)
    assert subs_on is None, f"形状校验失效，长条臂被当成菌落：{[(round(c.x), round(c.y), round(c.aspect, 1)) for c in (subs_on or [])]}"
    subs_off = _split_one(ctx, mask, dt, cand, dict(p, split_shape=False))
    assert subs_off is not None and len(subs_off) == 2, "关闭形状校验应回到旧行为（两根长条混入），证明开关生效"


def test_split_shape_toggle_default_on():
    assert default_params("adaptive")["split_shape"] is True


def test_split_fragments_must_pass_min_area():
    """回归（用户 E1-2 针尖菌落）：哑铃形小菌落拆出的碎片若低于 min_area，
    下游几何过滤会把碎片全灭、原轮廓又已被替换 → 菌落凭空消失。
    碎片下限须取 max(15%×候选, min_area)，此时保留原轮廓。"""
    import cv2

    from bac_count.core.models import Colony
    from bac_count.core.registry import PrepContext
    from bac_count.core.split import _split_one

    BROWN = (150, 120, 95)
    bgr = np.zeros((300, 300, 3), np.uint8)
    bgr[:, :] = (215, 195, 170)
    cv2.circle(bgr, (150, 150), 8, BROWN, -1)
    cv2.circle(bgr, (162, 150), 5, BROWN, -1)
    mask = np.zeros((300, 300), np.uint8)
    cv2.circle(mask, (150, 150), 8, 255, -1)
    cv2.circle(mask, (162, 150), 5, 255, -1)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ctx = PrepContext(bgr=bgr, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                      dish_mask=None, alpha=None)
    cand = Colony(x=155.5, y=150, r=12, area=250)
    p = default_params("adaptive")
    p["split_on"] = True

    subs_hi = _split_one(ctx, mask, dt, cand, dict(p, min_area=60))
    assert subs_hi is None, f"卫星碎片(≈45)<min_area=60 时拆分应整体作废：{subs_hi}"
    subs_lo = _split_one(ctx, mask, dt, cand, dict(p, min_area=30))
    assert subs_lo is not None and len(subs_lo) == 2, \
        f"min_area=30 时两碎片均可独立存活，拆分应生效：{subs_lo}"
