from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating
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
from explainaboard.processors.processor_registry import processor_registry
from explainaboard.utils.typing_utils import unwrap

sum_attr = SUMAttribute()


@processor_registry.register("summarization")
class SummarizationProcessor(ConditionalGenerationProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.summarization

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        f = super().default_analysis_levels()
        new_examp_features = {
            "sum_attributes": feature.Value(
                dtype="dict",
                func=lambda info, x, c: sum_attr.cal_attributes_each(
                    x["source"], x["reference"]
                ),
            ),
            "attr_compression": feature.Value(
                dtype="float",
                description="compression",
                func=lambda info, x, c: c.features['sum_attributes'][
                    "attr_compression"
                ],
            ),
            "attr_copy_len": feature.Value(
                dtype="float",
                description="copy length",
                func=lambda info, x, c: c.features['sum_attributes']["attr_copy_len"],
            ),
            "attr_coverage": feature.Value(
                dtype="float",
                description="coverage",
                func=lambda info, x, c: c.features['sum_attributes']["attr_coverage"],
            ),
            "attr_novelty": feature.Value(
                dtype="float",
                description="novelty",
                func=lambda info, x, c: c.features['sum_attributes']["attr_novelty"],
            ),
        }
        f[0].features.update(new_examp_features)
        return f

    @classmethod
    def _get_default_eaas_strs(cls):
        return ['rouge1', 'rouge2', 'rougeL', 'length_ratio']

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x['text'], unwrap(sys_info.source_tokenizer)
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x['summary'], unwrap(sys_info.target_tokenizer)
        )
        return {
            'source_vocab': source_vocab,
            'source_vocab_rank': source_vocab_rank,
            'target_vocab': target_vocab,
            'target_vocab_rank': target_vocab_rank,
        }
