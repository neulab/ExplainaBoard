"""A processor for the grammatical error correction task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import SeqCorrectCountConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor


class GrammaticalErrorCorrectionProcessor(Processor):
    """A processor for the grammatical error correction task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.grammatical_error_correction

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "text": feature.Value(dtype=feature.DataType.STRING),
            "edits": feature.Dict(
                feature={
                    "start_idx": feature.Sequence(
                        feature=feature.Value(dtype=feature.DataType.INT)
                    ),
                    "end_idx": feature.Sequence(
                        feature=feature.Value(dtype=feature.DataType.INT)
                    ),
                    "corrections": feature.Sequence(
                        feature=feature.Sequence(
                            feature=feature.Value(dtype=feature.DataType.STRING)
                        )
                    ),
                }
            ),
            "text_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="length of the text",
                func=lambda info, x, c: count_tokens(info, x["text"]),
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

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        return {}

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {"SeqCorrectCount": SeqCorrectCountConfig()}

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()

    def _get_true_label(self, data_point):
        """See Processor._get_true_label."""
        return data_point["edits"]

    def _get_predicted_label(self, data_point):
        """See Processor._get_predicted_label."""
        return data_point["predicted_edits"]
