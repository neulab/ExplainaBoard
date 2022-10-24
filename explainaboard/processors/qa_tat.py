"""A processor for the TAT-QA dataset."""

from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.qa_table_text_hybrid import (
    ExactMatchQATatConfig,
    F1ScoreQATatConfig,
)
from explainaboard.processors.processor import Processor
from explainaboard.utils.typing_utils import unwrap


class QATatProcessor(Processor):
    """A processor for the TAT-QA dataset."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.qa_tat

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features = {
            "question": feature.Value(dtype=feature.DataType.STRING),
            "context": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "table": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "true_answer": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "predicted_answer": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "predicted_answer_scale": feature.Value(dtype=feature.DataType.STRING),
            "answer_type": feature.Value(
                dtype=feature.DataType.STRING,
                description="type of answer",
            ),
            "answer_scale": feature.Value(
                dtype=feature.DataType.STRING,
                description="scale of answer",
            ),
            "context_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="context length in tokens",
                func=lambda info, x, c: sum(
                    [count_tokens(info, text) for text in x["context"]["text"]]
                ),
            ),
            "table_rows": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of table row",
                func=lambda info, x, c: len(x["table"]),
            ),
            "table_columns": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of table column",
                func=lambda info, x, c: len(x["table"][0])
                if len(x["table"]) > 0
                else 0,
            ),
            "question_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x["question"]),
            ),
            "answer_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the length of answer",
                func=lambda info, x, c: len(x["true_answer"]),
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
        features = self.default_analysis_levels()[0].features
        # Create analyses
        analyses: list[Analysis] = [
            BucketAnalysis(
                level="example",
                description=features["answer_type"].description,
                feature="answer_type",
                method="discrete",
                num_buckets=10,
            ),
            BucketAnalysis(
                level="example",
                description=features["answer_scale"].description,
                feature="answer_scale",
                method="discrete",
                num_buckets=10,
            ),
        ]
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {
            "ExactMatchQATat": ExactMatchQATatConfig(
                source_language=source_language,
                target_language=target_language,
            ),
            "F1ScoreQATat": F1ScoreQATatConfig(
                source_language=source_language,
                target_language=target_language,
            ),
        }

    def _get_true_label(self, data_point):
        """See processor._get_true_label."""
        return {
            "true_answer": data_point["true_answer"],
            "answer_type": data_point["answer_type"],
            "answer_scale": data_point["answer_scale"],
        }

    def _get_predicted_label(self, data_point):
        """See processor._get_predicted_label."""
        return {
            "predicted_answer": data_point["predicted_answer"],
            "predicted_answer_scale": data_point["predicted_answer_scale"],
        }

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x["question"], unwrap(sys_info.source_tokenizer)
        )

        return {"source_vocab": source_vocab, "source_vocab_rank": source_vocab_rank}
