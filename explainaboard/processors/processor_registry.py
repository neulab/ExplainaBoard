from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metric import MetricConfig
from explainaboard.processors.processor import Processor

_processor_registry: dict = {}


def get_processor(task: TaskType | str) -> Processor:
    """
    return a processor based on the task type
    TODO: error handling
    """
    task_cast: TaskType = TaskType(task)
    return _processor_registry[task_cast]()


def register_processor(task_type: TaskType):
    """
    a register for task specific processors.
    example usage: `@register_processor(TaskType.text_classification)`
    """

    def register_processor_fn(cls):
        _processor_registry[task_type] = cls
        return cls

    return register_processor_fn


def get_metric_list_for_processor(task: TaskType) -> list[MetricConfig]:
    processor_class = _processor_registry[task]
    return processor_class.full_metric_list()
