from __future__ import annotations

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.tabular_classification)
class TextClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.tabular_classification

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "true_label": feature.Value("string"),
                "predicted_label": feature.Value("string"),
                "label": feature.Value(
                    dtype="string",
                    description="category",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=4, setting=1
                    ),
                ),
            }
        )

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [AccuracyConfig(name='Accuracy')]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_label(self, sys_info: SysOutputInfo, existing_feature: dict):
        return existing_feature["true_label"]
