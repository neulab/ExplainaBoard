from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metric import BIOF1ScoreConfig, MetricConfig
from explainaboard.processors.processor_registry import register_processor
from explainaboard.processors.sequence_labeling import SeqLabProcessor
from explainaboard.utils.span_utils import BIOSpanOps


@register_processor(TaskType.named_entity_recognition)
class NERProcessor(SeqLabProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.named_entity_recognition

    @classmethod
    def default_metrics(cls, language=None) -> list[MetricConfig]:
        return [BIOF1ScoreConfig(name='F1', language=language)]

    @classmethod
    def default_span_ops(cls) -> BIOSpanOps:
        return BIOSpanOps()
