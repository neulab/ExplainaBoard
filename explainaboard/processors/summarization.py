"""A processor for the summarization task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datalabs.operations.featurize.plugins.summarization.sum_attribute import (
    SUMAttribute,
)

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import AnalysisLevel
from explainaboard.analysis.feature_funcs import accumulate_vocab_from_samples
from explainaboard.info import SysOutputInfo
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.utils.typing_utils import unwrap

sum_attr = SUMAttribute()


class SummarizationProcessor(ConditionalGenerationProcessor):
    """A processor for the summarization task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.summarization

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        f = super().default_analysis_levels()
        new_examp_features = {
            "sum_attributes": feature.Dict(
                feature={
                    "attr_density": feature.Value(dtype=feature.DataType.FLOAT),
                    "attr_coverage": feature.Value(dtype=feature.DataType.FLOAT),
                    "attr_compression": feature.Value(dtype=feature.DataType.FLOAT),
                    "attr_repetition": feature.Value(dtype=feature.DataType.FLOAT),
                    "attr_novelty": feature.Value(dtype=feature.DataType.FLOAT),
                    "attr_copy_len": feature.Value(dtype=feature.DataType.FLOAT),
                    "attr_source_len": feature.Value(dtype=feature.DataType.INT),
                    "attr_hypothesis_len": feature.Value(dtype=feature.DataType.INT),
                },
                func=lambda info, x, c: sum_attr.cal_attributes_each(
                    x["source"], x["reference"]
                ),
            ),
            "attr_compression": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="compression",
                func=lambda info, x, c: c.features["sum_attributes"][
                    "attr_compression"
                ],
            ),
            "attr_copy_len": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="copy length",
                func=lambda info, x, c: c.features["sum_attributes"]["attr_copy_len"],
            ),
            "attr_coverage": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="coverage",
                func=lambda info, x, c: c.features["sum_attributes"]["attr_coverage"],
            ),
            "attr_novelty": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="novelty",
                func=lambda info, x, c: c.features["sum_attributes"]["attr_novelty"],
            ),
        }
        f[0].features.update(new_examp_features)
        return f

    @classmethod
    def _get_default_eaas_strs(cls):
        return ["rouge1", "rouge2", "rougeL", "length_ratio"]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        samples_list = list(samples)
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples_list, lambda x: x["source"], unwrap(sys_info.source_tokenizer)
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples_list, lambda x: x["reference"], unwrap(sys_info.target_tokenizer)
        )
        return {
            "source_vocab": source_vocab,
            "source_vocab_rank": source_vocab_rank,
            "target_vocab": target_vocab,
            "target_vocab_rank": target_vocab_rank,
        }
