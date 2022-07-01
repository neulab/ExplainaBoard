from dataclasses import dataclass

from explainaboard.analysis.feature import Value
from explainaboard.analysis.analyses import BucketAnalysis


@dataclass
class AnalysisLevel:
    name: str
    features: dict[str, Value]
    analyses: [BucketAnalysis]