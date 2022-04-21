from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from explainaboard import TaskType
from explainaboard.constants import FileType, Source
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader import Loader

# loader_registry is a global variable, storing all basic loading functions
_loader_registry: dict[TaskType, type[Loader]] = {}


def get_custom_dataset_loader(
    task: TaskType | str,
    dataset_data: str,
    output_data: str,
    dataset_source: Source | None = None,
    output_source: Source | None = None,
    dataset_file_type: FileType | None = None,
    output_file_type: FileType | None = None,
) -> Loader:
    """returns a loader for a custom dataset"""
    task = TaskType(task)
    return _loader_registry[task](
        dataset_data,
        output_data,
        dataset_source,
        output_source,
        dataset_file_type,
        output_file_type,
    )


def get_datalab_loader(
    task: TaskType | str,
    dataset: DatalabLoaderOption,
    output_data: str,
    output_source: Optional[Source] = None,
    output_file_type: Optional[FileType] = None,
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


@dataclass
class SupportedFileFormats:
    custom_dataset: list[FileType] = field(default_factory=list)
    system_output: list[FileType] = field(default_factory=list)


def get_supported_file_types_for_loader(task: TaskType) -> SupportedFileFormats:
    loader_class = _loader_registry[task]
    return SupportedFileFormats(
        list(loader_class.default_dataset_file_loaders().keys()),
        list(loader_class.default_output_file_loaders().keys()),
    )
