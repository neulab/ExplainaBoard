from __future__ import annotations

from typing import Optional

from explainaboard import TaskType
from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import DatalabLoaderOption
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


def get_custom_dataset_loader(
    task: TaskType | str,
    dataset_data: str,
    output_data: str,
    dataset_source: Source | None = None,
    output_source: Source | None = None,
    dataset_file_type: FileType | None = None,
    output_file_type: FileType | None = None,
    field_mapping: dict[str, str] | None = None,
) -> Loader:
    """returns a loader for a custom dataset"""
    task = TaskType(task)
    return _loader_registry[task](
        dataset_data=dataset_data,
        output_data=output_data,
        dataset_source=dataset_source,
        output_source=output_source,
        dataset_file_type=dataset_file_type,
        output_file_type=output_file_type,
        field_mapping=field_mapping,
    )


def get_datalab_loader(
    task: TaskType | str,
    dataset: DatalabLoaderOption,
    output_data: str,
    output_source: Optional[Source] = None,
    output_file_type: Optional[FileType] = None,
    field_mapping: dict[str, str] | None = None,
) -> Loader:
    """uses a loader for a dataset from datalab. The loader downloads the dataset
    and merges the user provided output with the dataset"""
    task = TaskType(task)
    return _loader_registry[task](
        dataset_data=dataset,
        output_data=output_data,
        dataset_source=Source.in_memory,
        output_source=output_source,
        dataset_file_type=FileType.datalab,
        output_file_type=output_file_type,
        field_mapping=field_mapping,
    )


def register_loader(task_type: TaskType):
    """
    a register for different data loaders, for example
    For example, `@register_loader(TaskType.text_classification)`
    """

    def register_loader_fn(cls):
        _loader_registry[task_type] = cls
        return cls

    return register_loader_fn
