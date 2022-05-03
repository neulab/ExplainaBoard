from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metric import BMESF1ScoreConfig, MetricConfig
from explainaboard.processors.processor_registry import register_processor
from explainaboard.processors.sequence_labeling import SeqLabProcessor
from explainaboard.utils.span_utils import BMESSpanOps


@register_processor(TaskType.word_segmentation)
class CWSProcessor(SeqLabProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.word_segmentation

    @classmethod
    def default_metrics(cls, language=None) -> list[MetricConfig]:
        return [BMESF1ScoreConfig(name='F1', language=language)]

    @classmethod
    def default_span_ops(cls) -> BMESSpanOps:
        return BMESSpanOps()
