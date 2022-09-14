"""A registry for Loader classes to look them up by class name."""
from __future__ import annotations

from explainaboard import TaskType
from explainaboard.loaders.loader import Loader

# loader_registry is a global variable, storing all basic loading functions
_loader_registry: dict[TaskType, type[Loader]] = {}


def get_loader_class(task: TaskType | str) -> type[Loader]:
    """Obtains the loader class for the specified task type.

    Args:
        task: Task type or task name.

    Returns:
        The Loader class associated to `task`.
    """
    return _loader_registry[TaskType(task)]


def register_loader(task_type: TaskType):
    """A register for different data loaders.

    For example, `@register_loader(TaskType.text_classification)`
    """

    def register_loader_fn(cls):
        _loader_registry[task_type] = cls
        return cls

    return register_loader_fn
