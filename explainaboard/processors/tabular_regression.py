from __future__ import annotations

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
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

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "true_value": feature.Value("float"),
                "predicted_value": feature.Value("float"),
                "value": feature.Value(
                    dtype="float",
                    description="predicted value",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=None,
                    ),
                ),
            }
        )

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            RootMeanSquaredErrorConfig(name='RMSE'),
            AbsoluteErrorConfig(name='AbsoluteError'),
        ]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_value(self, sys_info: SysOutputInfo, existing_feature: dict):
        return existing_feature["true_value"]

    def _get_true_label(self, data_point):
        return data_point["true_value"]

    def _get_predicted_label(self, data_point):
        return data_point["predicted_value"]
