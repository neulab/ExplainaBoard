"""Package definition for explainaboard."""

from __future__ import annotations

from explainaboard.constants import FileType, Source, TaskType
from explainaboard.loaders import DatalabLoaderOption, get_loader_class
from explainaboard.meta_analyses import RankFlippingMetaAnalysis, RankingMetaAnalysis
from explainaboard.processors import get_processor_class

__all__ = [
    "FileType",
    "DatalabLoaderOption",
    "get_loader_class",
    "get_processor_class",
    "Source",
    "TaskType",
    "RankFlippingMetaAnalysis",
    "RankingMetaAnalysis",
]
