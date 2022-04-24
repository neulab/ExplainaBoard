from __future__ import annotations

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import EaaSMetricConfig, MetricConfig
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.machine_translation)
class MachineTranslationProcessor(ConditionalGenerationProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.machine_translation

    @classmethod
    def default_features(cls) -> feature.Features:
        f = super().default_features()
        f.update(
            feature.Features(
                {
                    # declaim task-specific features
                    "attr_compression": feature.Value(
                        dtype="float",
                        description="the ratio between source and reference length",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                }
            )
        )
        return f

    @classmethod
    def default_metrics(cls, language=None) -> list[MetricConfig]:
        return [EaaSMetricConfig(name='bleu', language=language)]

    def _get_attr_compression(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(sys_info.tokenize(existing_features["source"])) / len(
            sys_info.tokenize(existing_features["reference"])
        )
