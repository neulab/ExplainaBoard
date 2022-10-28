"""A processor for the argument pair identification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import (
    count_tokens,
    feat_freq_rank,
    feat_length_freq,
    feat_num_oov,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap


class ArgumentPairIdentificationProcessor(Processor):
    """A processor for the argument pair identification task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.argument_pair_identification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "context": feature.Value(dtype=feature.DataType.STRING),
            "query": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "true_label": feature.Value(dtype=feature.DataType.STRING),
            "predicted_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="the predicted label",
            ),
            "context_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x["context"]),
            ),
            "query_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the length in tokens of true query",
                func=lambda info, x, c: count_tokens(
                    info, x["query"][int(x["true_label"])]
                ),
            ),
            "num_oov": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x["context"], stat["vocab"]
                ),
            ),
            "fre_rank": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x["context"], stat["vocab_rank"]
                ),
            ),
            "length_fre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the frequency of context length in training set",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_length_freq(
                    info, x["context"], stat["length_fre"]
                ),
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
        analyses: list[Analysis] = []
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    @classmethod
    def default_metrics(
        cls, level="example", source_language=None, target_language=None
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {"Accuracy": AccuracyConfig()}

    def _statistics_func(
        self, samples: Iterable[Any], sys_info: SysOutputInfo
    ) -> dict[str, Any]:
        """See Processor._statistics_func."""
        vocab: Counter = Counter()
        length_counts: Counter = Counter()

        total_samps = 0
        tokenizer = unwrap(sys_info.source_tokenizer)
        for sample in progress(samples):
            text = sample["context"]
            tokens = tokenizer(text)
            length = len(tokens)

            length_counts[length] += 1

            # update vocabulary
            for w in tokens:
                vocab[w] += 1

            total_samps += 1

        # the rank of each word based on its frequency
        vocab_uniq_counts = sorted(set(vocab.values()), reverse=True)
        vocab_count_ranks = {
            count: rank for rank, count in enumerate(vocab_uniq_counts, 1)
        }
        vocab_rank = {k: vocab_count_ranks[count] for k, count in vocab.items()}

        length_fre = {k: v / total_samps for k, v in length_counts.items()}

        return {"vocab": vocab, "vocab_rank": vocab_rank, "length_fre": length_fre}
