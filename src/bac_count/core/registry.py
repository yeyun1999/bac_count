"""算法注册表：新增检测算法 = 实现 Detector + @register。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bac_count.core.models import Colony, ParamSpec

__all__ = ["PrepContext", "Detector", "register", "DETECTORS", "get_detector"]


@dataclass
class PrepContext:
    """预处理产物（均为工作图坐标系）。"""

    bgr: np.ndarray  # 工作尺寸 BGR
    gray: np.ndarray  # 通道选择+阴影校正+模糊+CLAHE+极性统一后的 uint8，菌落恒为暗色
    dish_mask: np.ndarray | None  # uint8 0/255，皿内可检测区域
    alpha: np.ndarray | None  # 原始 alpha 通道（工作尺寸）或 None
    # 皿圆（工作图坐标）。run_detection 产出的 ctx 里这个字段**总是已求值**：
    # None = 这张图没找到皿。调用方可按图缓存它，下次经
    # run_detection(dish_circle=...) 传回，跳过最贵的 Hough 找圆（见 preprocess.find_dish_circle）。
    dish_circle: tuple[tuple[float, float], float] | None = None


class Detector(ABC):
    name: str = ""
    display: str = ""

    def schema(self) -> list[ParamSpec]:
        return []

    @abstractmethod
    def detect(self, ctx: PrepContext, params: dict[str, Any]) -> list[Colony]:
        """返回候选菌落（工作图坐标），后处理过滤由 pipeline 统一执行。"""


DETECTORS: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    DETECTORS[cls.name] = cls
    return cls


def get_detector(name: str) -> Detector:
    return DETECTORS[name]()
