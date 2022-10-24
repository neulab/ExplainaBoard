"""A processor for the machine translation task."""

from __future__ import annotations

from collections.abc import Iterable
import copy
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import AnalysisLevel
from explainaboard.analysis.feature_funcs import accumulate_vocab_from_samples
from explainaboard.info import SysOutputInfo
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.utils.typing_utils import unwrap


class MachineTranslationProcessor(ConditionalGenerationProcessor):
    """A processor for the machine translation task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.machine_translation

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        f = super().default_analysis_levels()
        f = copy.deepcopy(f)
        f[0].features["attr_compression"] = feature.Value(
            dtype=feature.DataType.FLOAT,
            description="the ratio between source and reference length",
            func=lambda info, x, c: c.features["source_length"]
            / c.features["reference_length"],
        )

        return f

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        if sys_info.source_language is None or sys_info.target_language is None:
            raise ValueError(
                "source or target languages must be specified to load "
                f"translation data, but source={sys_info.source_language} "
                f", target={sys_info.target_language}"
            )

        samples_list = list(samples)

        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples_list,
            lambda x: x["source"],
            unwrap(sys_info.source_tokenizer),
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples_list,
            lambda x: x["reference"],
            unwrap(sys_info.target_tokenizer),
        )
        return {
            "source_vocab": source_vocab,
            "source_vocab_rank": source_vocab_rank,
            "target_vocab": target_vocab,
            "target_vocab_rank": target_vocab_rank,
        }
