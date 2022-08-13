from __future__ import annotations

from collections.abc import Iterator
import copy

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import AnalysisLevel
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

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        f = super().default_analysis_levels()
        f = copy.deepcopy(f)
        f[0].features["attr_compression"] = feature.Value(
            dtype="float",
            description="the ratio between source and reference length",
            func=lambda info, x, c: c.features['source_length']
            / c.features['reference_length'],
        )

        return f

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

        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples,
            lambda x: FileLoader.find_field(x, src),
            unwrap(sys_info.source_tokenizer),
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples,
            lambda x: FileLoader.find_field(x, trg),
            unwrap(sys_info.target_tokenizer),
        )
        return {
            'source_vocab': source_vocab,
            'source_vocab_rank': source_vocab_rank,
            'target_vocab': target_vocab,
            'target_vocab_rank': target_vocab_rank,
        }
