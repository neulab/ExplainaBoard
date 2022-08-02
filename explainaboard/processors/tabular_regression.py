from __future__ import annotations

from typing import cast, List

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature import FeatureType
from explainaboard.metrics.continuous import (
    AbsoluteErrorConfig,
    RootMeanSquaredErrorConfig,
)
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.tabular_regression)
class TabularRegressionProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.tabular_regression

    def default_analyses(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "true_value": feature.Value(
                dtype="float",
                description="the true value of the input",
            ),
            "predicted_value": feature.Value(
                dtype="float",
                description="the predicted value",
            ),
        }
        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        # Create analyses
        analyses: list[Analysis] = []
        for x in continuous_features:
            analyses.append(
                BucketAnalysis(
                    description=features[x].description, feature=x, method="continuous"
                )
            )

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
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            RootMeanSquaredErrorConfig(name='RMSE'),
            AbsoluteErrorConfig(name='AbsoluteError'),
        ]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_true_label(self, data_point):
        return data_point["true_value"]

    def _get_predicted_label(self, data_point):
        return data_point["predicted_value"]
