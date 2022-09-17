"""A registry for processors."""

from __future__ import annotations

from explainaboard import TaskType
from explainaboard.processors.processor import Processor

_processor_registry: dict = {}


def get_processor(task: TaskType | str) -> Processor:
    """Return a processor based on the task type.

    TODO: error handling

    Args:
        task: The type of task

    Returns:
        The processor for that task
    """
    task_cast: TaskType = TaskType(task)
    return _processor_registry[task_cast]()


def register_processor(task_type: TaskType):
    """A register for task specific processors.

    example usage: `@register_processor(TaskType.text_classification)`
    """

    def register_processor_fn(cls):
        """The function to register."""
        _processor_registry[task_type] = cls
        return cls

    return register_processor_fn
