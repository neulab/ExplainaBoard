from __future__ import annotations

from explainaboard import TaskType
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.serialization.registry import TypeRegistry

processor_registry = TypeRegistry[Processor]()


def get_processor(task: TaskType | str) -> Processor:
    """
    return a processor based on the task type
    TODO: error handling
    """
    if isinstance(task, TaskType):
        name = task.name
    elif isinstance(task, str):
        name = task
    processor_class = processor_registry.get_type(name)
    return processor_class()


def get_metric_list_for_processor(task: TaskType) -> list[MetricConfig]:
    processor_class = processor_registry.get_type(str(task))
    return processor_class.full_metric_list()
