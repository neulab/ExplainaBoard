from typing import Iterable
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor


_processor_registry: dict = {}


def get_processor(task: TaskType, metadata: dict = None, data: Iterable[dict] = None) -> Processor:
    """
    return a processor based on the task type
    TODO: error handling
    """
    return _processor_registry[task](metadata, data)


def register_processor(task_type: TaskType):
    """
    a register for task specific processors. 
    example usage: `@register_processor(TaskType.text_classification)`
    """
    def register_processor_fn(cls):
        _processor_registry[task_type] = cls
        return cls
    return register_processor_fn
