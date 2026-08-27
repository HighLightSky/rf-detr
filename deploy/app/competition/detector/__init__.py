"""检测器后端模块。"""

from competition.detector.base import Detector
from competition.detector.multi_backend import MultiBackendDetector

__all__ = ["Detector", "MultiBackendDetector"]
