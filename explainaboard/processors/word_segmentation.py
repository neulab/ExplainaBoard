"""A processor for the word segmentation task."""

from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metrics.f1_score import F1ScoreConfig, SeqF1ScoreConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor_registry import register_processor
from explainaboard.processors.sequence_labeling import SeqLabProcessor
from explainaboard.utils.span_utils import BMESSpanOps


@register_processor(TaskType.word_segmentation)
class CWSProcessor(SeqLabProcessor):
    """A processor for the word segmentation task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.word_segmentation

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """See Processor.default_metrics."""
        defaults: dict[str, list[MetricConfig]] = {
            'example': [
                SeqF1ScoreConfig(
                    name='F1',
                    source_language=source_language,
                    target_language=target_language,
                    tag_schema='bmes',
                )
            ],
            'span': [
                F1ScoreConfig(
                    name='F1',
                    source_language=source_language,
                    target_language=target_language,
                    ignore_classes=[cls._DEFAULT_TAG],
                )
            ],
        }
        return defaults[level]

    @classmethod
    def _default_span_ops(cls) -> BMESSpanOps:
        return BMESSpanOps()
