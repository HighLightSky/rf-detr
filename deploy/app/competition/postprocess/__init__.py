"""比赛结果后处理模块。"""

from competition.postprocess.base import Postprocessor
from competition.postprocess.shwx_competition import ShwxCompetitionPostprocessor

__all__ = ["Postprocessor", "ShwxCompetitionPostprocessor"]
