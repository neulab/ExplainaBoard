from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metric import MetricConfig, SeqF1ScoreConfig
from explainaboard.processors.processor_registry import register_processor
from explainaboard.processors.sequence_labeling import SeqLabProcessor
from explainaboard.utils.span_utils import BIOSpanOps


@register_processor(TaskType.chunking)
class ChunkingProcessor(SeqLabProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.chunking

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            SeqF1ScoreConfig(
                name='F1',
                source_language=source_language,
                target_language=target_language,
                tag_schema='bio',
            )
        ]

    @classmethod
    def default_span_ops(cls) -> BIOSpanOps:
        return BIOSpanOps()
