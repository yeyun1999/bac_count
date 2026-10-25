"""Unicode 路径安全的图像读写。

Windows 上 cv2.imread/imwrite 用进程 ANSI 代码页解析路径：Python 传入的中文
str 经 UTF-8 字节被按 GBK 误解码（课题→璇鹃），文件直接找不到。
Python 自带的 read_bytes/write_bytes 走宽字符 API 无此问题——先读字节再
cv2.imdecode、先 cv2.imencode 再写字节，即可绕开 OpenCV 的路径处理。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread_any", "imwrite_any"]


def imread_any(path: str | Path) -> np.ndarray | None:
    """等价 cv2.imread(IMREAD_UNCHANGED)；失败返回 None（不抛异常，由调用方判空）。"""
    try:
        buf = np.frombuffer(Path(path).read_bytes(), np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def imwrite_any(path: str | Path, img: np.ndarray) -> None:
    """等价 cv2.imwrite；扩展名取自 path（默认 .png）。"""
    p = Path(path)
    ext = p.suffix.lower() if p.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图像编码失败: {path}")
    p.write_bytes(buf.tobytes())
