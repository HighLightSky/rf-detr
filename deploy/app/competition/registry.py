"""受控模块注册表。"""

from __future__ import annotations

from competition.config import PostprocessConfig, PreprocessConfig
from competition.postprocess.base import Postprocessor
from competition.postprocess.shwx_competition import ShwxCompetitionPostprocessor
from competition.preprocess.base import Preprocessor
from competition.preprocess.direct import DirectPreprocessor
from competition.preprocess.shwx_large_image import ShwxLargeImagePreprocessor


def build_preprocessor(config: PreprocessConfig) -> Preprocessor:
    """通过白名单实例化图像预处理模块。"""
    if config.name == "direct":
        return DirectPreprocessor()
    if config.name == "shwx_large_image":
        return ShwxLargeImagePreprocessor(config)
    raise ValueError(f"未注册的预处理模块: {config.name}")


def build_postprocessor(config: PostprocessConfig) -> Postprocessor:
    """通过白名单实例化比赛输出后处理模块。"""
    if config.name == "shwx_competition_v1":
        return ShwxCompetitionPostprocessor(config)
    raise ValueError(f"未注册的后处理模块: {config.name}")
