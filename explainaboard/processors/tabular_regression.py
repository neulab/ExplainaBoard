"""A processor for the tabular regression task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.continuous import (
    AbsoluteErrorConfig,
    RootMeanSquaredErrorConfig,
)
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor


class TabularRegressionProcessor(Processor):
    """A processor for the tabular regression task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.tabular_regression

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "true_value": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the true value of the input",
            ),
            "predicted_value": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the predicted value",
            ),
        }

        return [
            AnalysisLevel(
                name="example",
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        """See Processor.default_analyses."""
        return self.continuous_feature_analyses()

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {
            "RMSE": RootMeanSquaredErrorConfig(),
            "AbsoulteError": AbsoluteErrorConfig(),
        }

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        return {}

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_true_label(self, data_point):
        """See processor._get_true_label."""
        return data_point["true_value"]

    def _get_predicted_label(self, data_point):
        """See processor._get_predicted_label."""
        return data_point["predicted_value"]
