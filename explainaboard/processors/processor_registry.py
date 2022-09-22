"""A registry for processors."""

from __future__ import annotations

from explainaboard import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.serialization.registry import TypeRegistry

processor_registry = TypeRegistry[Processor]()


def get_processor(task: TaskType | str) -> Processor:
    """Return a processor based on the task type.

    TODO: error handling

    Args:
        task: The type of task

    Returns:
        The processor for that task
    """
    processor_class = processor_registry.get_type(task.name)
    return processor_class()


def get_metric_list_for_processor(task: TaskType) -> list[MetricConfig]:
    processor_class = processor_registry.get_type(str(task))
    return processor_class.full_metric_list()
