"""A processor for the word segmentation task."""

from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metrics.f1_score import F1ScoreConfig, SeqF1ScoreConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.sequence_labeling import SeqLabProcessor
from explainaboard.utils.span_utils import BMESSpanOps


class CWSProcessor(SeqLabProcessor):
    """A processor for the word segmentation task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.word_segmentation

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        defaults: dict[str, dict[str, MetricConfig]] = {
            "example": {
                "F1": SeqF1ScoreConfig(
                    source_language=source_language,
                    target_language=target_language,
                    tag_schema="bmes",
                )
            },
            "span": {
                "F1": F1ScoreConfig(
                    source_language=source_language,
                    target_language=target_language,
                    ignore_classes=[cls._DEFAULT_TAG],
                )
            },
        }
        return defaults[level]

    @classmethod
    def _default_span_ops(cls) -> BMESSpanOps:
        return BMESSpanOps()
