from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.loaders.file_loader import FileLoader, FileLoaderField
from explainaboard.metric import EaaSMetricConfig, MetricConfig
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.feature_funcs import accumulate_vocab_from_samples
from explainaboard.utils.typing_utils import unwrap


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
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            EaaSMetricConfig(
                name='bleu',
                source_language=source_language,
                target_language=target_language,
            ),
            EaaSMetricConfig(
                name='length_ratio',
                source_language=source_language,
                target_language=target_language,
            ),
        ]

    def _get_attr_compression(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(
            unwrap(sys_info.source_tokenizer)(existing_features["source"])
        ) / len(unwrap(sys_info.target_tokenizer)(existing_features["reference"]))

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        if sys_info.source_language is None or sys_info.target_language is None:
            raise ValueError(
                'source or target languages must be specified to load '
                f'translation data, but source={sys_info.source_language} '
                f', target={sys_info.target_language}'
            )
        src = FileLoaderField(('translation', sys_info.source_language), '', str)
        trg = FileLoaderField(('translation', sys_info.target_language), '', str)
        return {
            'source_vocab': accumulate_vocab_from_samples(
                samples,
                lambda x: FileLoader.find_field(x, src),
                unwrap(sys_info.source_tokenizer),
            ),
            'target_vocab': accumulate_vocab_from_samples(
                samples,
                lambda x: FileLoader.find_field(x, trg),
                unwrap(sys_info.target_tokenizer),
            ),
        }
