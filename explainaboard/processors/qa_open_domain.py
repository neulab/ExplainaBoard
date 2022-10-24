"""A processor for the open-domain QA task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.extractive_qa import ExactMatchQAConfig, F1ScoreQAConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.typing_utils import unwrap


class QAOpenDomainProcessor(Processor):
    """A processor for the open-domain QA task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.qa_open_domain

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features = {
            "question": feature.Value(dtype=feature.DataType.STRING),
            # "question_types": feature.Sequence(feature=feature.Value("string")),
            "answers": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "question_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x["question"]),
            ),
            "answer_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(
                    info, x["answers"][0], side="target"
                ),
            ),
            "num_oov": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of out-of-vocabulary words in the context",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x["context"], stat["source_vocab"]
                ),
            ),
            "fre_rank": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=(
                    "average rank of context words based on training set freq"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x["context"], stat["source_vocab_rank"]
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
        return self.continuous_feature_analyses()

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {
            "ExactMatch": ExactMatchQAConfig(
                source_language=source_language,
                target_language=target_language,
            ),
            "F1": F1ScoreQAConfig(
                source_language=source_language,
                target_language=target_language,
            ),
        }

    def _get_true_label(self, data_point):
        """See processor._get_true_label."""
        return data_point["answers"]

    def _get_predicted_label(self, data_point):
        """See processor._get_predicted_label."""
        return data_point["predicted_answer"]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x["question"], unwrap(sys_info.source_tokenizer)
        )

        return {"source_vocab": source_vocab, "source_vocab_rank": source_vocab_rank}
