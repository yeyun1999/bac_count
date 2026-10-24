"""后台检测线程：GUI 通过 submit 信号投递任务，done 信号回传结果。"""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from bac_count.core.models import Colony
from bac_count.core.pipeline import run_detection


class DetectWorker(QObject):
    """放在独立 QThread 中；submit 从 GUI 线程发射，槽在工作线程执行。"""

    submit = Signal(int, object, object, object)  # seq, bgr, alpha, params
    done = Signal(int, object)                    # seq, list[Colony]

    def __init__(self):
        super().__init__()
        self.submit.connect(self._run)

    @Slot(int, object, object, object)
    def _run(self, seq: int, bgr, alpha, params: dict) -> None:
        try:
            colonies, _ctx, _scale = run_detection(bgr, alpha, params)
        except Exception:
            traceback.print_exc()
            colonies = []
        self.done.emit(seq, colonies)


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
