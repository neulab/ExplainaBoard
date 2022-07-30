from __future__ import annotations

from typing import cast, List

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    BucketAnalysis,
    ComboCountAnalysis,
)
from explainaboard.analysis.feature import FeatureType
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.tabular_classification)
class TextClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.tabular_classification

    def default_analyses(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "true_label": feature.Value(
                dtype="string",
                description="the true label of the input",
            ),
            "predicted_label": feature.Value(
                dtype="string",
                description="the predicted label",
            ),
        }
        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        # Create analyses
        analyses: list[Analysis] = [
            BucketAnalysis(
                feature="true_label",
                method="discrete",
                number=15,
            ),
            ComboCountAnalysis(features=("true_label", "predicted_label")),
        ]
        for x in continuous_features:
            analyses.append(BucketAnalysis(x, method="continuous"))

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=self.default_metrics(),
                analyses=cast(List[Analysis], analyses),
            )
        ]

    @classmethod
    def default_metrics(
        cls, level="example", source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [AccuracyConfig(name='Accuracy')]
