from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature_funcs import accumulate_vocab_from_samples
from explainaboard.info import SysOutputInfo
from explainaboard.loaders.file_loader import FileLoader, FileLoaderField
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.machine_translation)
class MachineTranslationProcessor(ConditionalGenerationProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.machine_translation

    def default_analyses(self) -> list[AnalysisLevel]:
        f = super().default_analyses()
        f[0].features["attr_compression"] = feature.Value(
            dtype="float",
            description="the ratio between source and reference length",
            func=lambda info, x, c: c.features['source_length']
            / c.features['reference_length'],
        )
        f[0].analyses.append(BucketAnalysis('attr_compression', method="continuous"))

        return f

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
