from dataclasses import dataclass

from explainaboard.analysis.feature import Value
from explainaboard.analysis.analyses import BucketAnalysis
from explainaboard.metrics.metric import Metric


@dataclass
class AnalysisLevel:
    name: str
    features: dict[str, Value]
    analyses: list[BucketAnalysis]
    metrics: list[Metric]