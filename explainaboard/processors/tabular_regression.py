from __future__ import annotations

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.metrics.continuous import (
    AbsoluteErrorConfig,
    RootMeanSquaredErrorConfig,
)
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.tabular_regression)
class TabularRegressionProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.tabular_regression

    def default_analysis_levels(self) -> list[AnalysisLevel]:
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

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        return self.continuous_feature_analyses()

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
