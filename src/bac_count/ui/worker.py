"""后台检测线程：GUI 通过 submit 信号投递任务，done 信号回传结果。"""
from __future__ import annotations

import traceback

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

from bac_count.core.models import Colony
from bac_count.core.pipeline import run_detection


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

    submit = Signal(int, object, object, object)  # seq, bgr, alpha, params
    done = Signal(int, object, object)            # seq, list[Colony], prep_display

    def __init__(self):
        super().__init__()
        self.submit.connect(self._run)

    @Slot(int, object, object, object)
    def _run(self, seq: int, bgr, alpha, params: dict) -> None:
        prep = None
        try:
            colonies, ctx, _scale = run_detection(bgr, alpha, params)
            prep = _prep_display(ctx)
        except Exception:
            traceback.print_exc()
            colonies = []
        self.done.emit(seq, colonies, prep)


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
