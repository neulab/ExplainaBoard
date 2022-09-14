"""A processor for the tabular classification task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    BucketAnalysis,
    ComboCountAnalysis,
)
from explainaboard.analysis.feature import FeatureType
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.tabular_classification)
class TextClassificationProcessor(Processor):
    """A processor for the tabular classification task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.tabular_classification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "true_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="the true label of the input",
            ),
            "predicted_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="the predicted label",
            ),
        }

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        """See Processor.default_analyses."""
        features = self.default_analysis_levels()[0].features
        # Create analyses
        analyses: list[Analysis] = [
            BucketAnalysis(
                level="example",
                description=features["true_label"].description,
                feature="true_label",
                method="discrete",
                number=15,
            ),
            ComboCountAnalysis(
                level="example",
                description="confusion matrix",
                features=("true_label", "predicted_label"),
            ),
        ]
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        return {}

    @classmethod
    def default_metrics(
        cls, level="example", source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """See Processor.default_metrics."""
        return [AccuracyConfig(name='Accuracy')]
