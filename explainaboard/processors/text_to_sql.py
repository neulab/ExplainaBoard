"""A processor for the text-to-SQL task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.text_to_sql import SQLExactSetMatchConfig, SQLExecutionConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.typing_utils import unwrap


class TextToSQLProcessor(Processor):
    """A processor for the text-to-SQL task."""

    SQL_KEYWORDS = [
        "select",
        "join",
        "where",
        "group",
        "sort",
        "having",
        "order",
        "asc",
        "desc",
        "limit",
    ]

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.text_to_sql

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels.

        Note: We use SQLite to evaluate SQLs, where SQL keywords and
         table/column names are case-insensitive.
        """
        features: dict[str, FeatureType] = {
            "question": feature.Value(
                dtype=feature.DataType.STRING,
                description="the input question",
            ),
            "true_sql": feature.Value(
                dtype=feature.DataType.STRING,
                description="the true sql",
            ),
            "predicted_sql": feature.Value(
                dtype=feature.DataType.STRING,
                description="the predicted sql",
            ),
            "db_id": feature.Value(
                dtype=feature.DataType.STRING,
                description="the database id",
            ),
            "question_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="input question length in tokens",
                func=lambda info, x, c: count_tokens(info, x["question"]),
            ),
            "sql_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x["true_sql"]),
            ),
        }

        # TODO(shuaichen): parse the sql query to make the detection robust
        for keyword in self.SQL_KEYWORDS:
            features[f"contains_{keyword}"] = feature.Value(
                dtype=feature.DataType.STRING,
                description=f"whether the SQL contains keyword '{keyword}'",
                func=lambda info, x, c, keyword=keyword: "yes"
                if f"{keyword}" in x["true_sql"].lower().split()
                else "no",
            )
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
        analyses: list[Analysis] = []
        for keyword in self.SQL_KEYWORDS:
            analyses.append(
                BucketAnalysis(
                    level="example",
                    description=features[f"contains_{keyword}"].description,
                    feature=f"contains_{keyword}",
                    method="discrete",
                    num_buckets=15,
                )
            )

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
            "ExactSetMatch": SQLExactSetMatchConfig(
                source_language=source_language,
                target_language=target_language,
            ),
            "Execution": SQLExecutionConfig(
                source_language=source_language,
                target_language=target_language,
            ),
        }

    def _get_true_label(self, data_point: dict):
        """Override the parent class.

        Return a pair of a SQL query and the database it corresponds to.
        See processor._get_true_label for more details.
        """
        return [data_point["true_sql"], data_point["db_id"]]

    def _get_predicted_label(self, data_point: dict):
        """Override the parent class.

        Return a pair of a SQL query and the database it corresponds to.
        See processor._get_predicted_label for more details.
        """
        return [
            data_point["predicted_sql"].split("\t")[0],
            data_point["predicted_sql"].split("\t")[1],
        ]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo) -> Any:
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x["question"], unwrap(sys_info.source_tokenizer)
        )

        return {"source_vocab": source_vocab, "source_vocab_rank": source_vocab_rank}
