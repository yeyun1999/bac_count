"""后台检测线程：GUI 通过 submit 信号投递任务，done 信号回传结果。"""
from __future__ import annotations

import traceback
from typing import Any

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

from bac_count.core.models import Colony
from bac_count.core.pipeline import UNSET, run_detection


def _prep_display(ctx) -> "object":
    """预处理灰度图 → BGR，并把皿 ROI 边界画上去（方便调 rim_pct/阴影等）。"""
    img = cv2.cvtColor(ctx.gray, cv2.COLOR_GRAY2BGR)
    if ctx.dish_mask is not None:
        cnts, _ = cv2.findContours(ctx.dish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(img, max(cnts, key=cv2.contourArea), -1, (0, 80, 255), 1)
    return img


class DetectWorker(QObject):
    """放在独立 QThread 中；submit 从 GUI 线程发射，槽在工作线程执行。"""

    # key = 图片路径字符串：done 原样带回，GUI 侧据此判断结果属于哪张图（防串图）
    submit = Signal(int, str, object, object, object)  # seq, key, bgr, alpha, params
    done = Signal(int, str, object, object)            # seq, key, list[Colony], prep_display

    def __init__(self):
        super().__init__()
        self.submit.connect(self._run)
        # 皿圆单槽缓存：无 alpha 的图每次检测都要跑 Hough 找圆（1200px 实测 ~173ms，
        # 占整条流水线 71%），而它只依赖图像本身、不随 rim_pct/其他参数变。
        # 键用数组**身份**（同一张图的 self.bgr 对象在拖滑杆期间不变）并持强引用，
        # 所以只缓存一张、换图即替换——绝不要用模块级 lru_cache（ndarray 不可哈希，
        # 且会把整图内存跨图留住）。
        self._img: object = None
        self._circle: Any = UNSET

    @Slot(int, str, object, object, object)
    def _run(self, seq: int, key: str, bgr, alpha, params: dict) -> None:
        prep = None
        try:
            circle = self._circle if bgr is self._img else UNSET
            colonies, ctx, _scale = run_detection(bgr, alpha, params, dish_circle=circle)
            self._img, self._circle = bgr, ctx.dish_circle  # ctx 里的值总是已求值（None=没找到皿）
            prep = _prep_display(ctx)
        except Exception:
            traceback.print_exc()
            colonies = []
        self.done.emit(seq, key, colonies, prep)


class DetectionService:
    """线程生命周期管理。"""

    def __init__(self, parent=None):
        self.thread = QThread(parent)
        self.worker = DetectWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()

    def stop(self) -> None:
        self.thread.quit()
        self.thread.wait(2000)
