"""图像预处理模块。"""

from competition.preprocess.base import Preprocessor
from competition.preprocess.direct import DirectPreprocessor
from competition.preprocess.shwx_large_image import ShwxLargeImagePreprocessor

__all__ = ["DirectPreprocessor", "Preprocessor", "ShwxLargeImagePreprocessor"]
