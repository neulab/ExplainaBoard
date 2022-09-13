"""Loaders for conditional generation tasks."""

from __future__ import annotations

from explainaboard.constants import FileType, TaskType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.conditional_generation)
@register_loader(TaskType.machine_translation)
class ConditionalGenerationLoader(Loader):
    """Loader for the conditional generation task.

    usage:
        please refer to `test_loaders.py`
    """

    OUTPUT_FIELDS = ['source', 'reference']
    JSON_FIELDS: list[str | tuple[str, str]] = ['source', 'reference']
    JSON_FIELDS_DATALAB: list[str | tuple[str, str]] = [
        'source_column',
        'reference_column',
    ]

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        """See Loader.default_dataset_file_type."""
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_dataset_file_loaders."""
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, cls.OUTPUT_FIELDS[0], str),
                    FileLoaderField(1, cls.OUTPUT_FIELDS[1], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField(cls.JSON_FIELDS[0], cls.OUTPUT_FIELDS[0], str),
                    FileLoaderField(cls.JSON_FIELDS[1], cls.OUTPUT_FIELDS[1], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField(
                        cls.JSON_FIELDS_DATALAB[0], cls.OUTPUT_FIELDS[0], str
                    ),
                    FileLoaderField(
                        cls.JSON_FIELDS_DATALAB[1], cls.OUTPUT_FIELDS[1], str
                    ),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        """See Loader.default_output_file_loaders."""
        field_name = "hypothesis"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, str)]
            ),
        }


@register_loader(TaskType.summarization)
class SummarizationLoader(ConditionalGenerationLoader):
    """A loader for summarization."""

    JSON_FIELDS_DATALAB: list[str | tuple[str, str]] = [
        'source_column',
        'reference_column',
    ]


@register_loader(TaskType.machine_translation)
class MachineTranslationLoader(ConditionalGenerationLoader):
    """A loader for machine translation."""

    JSON_FIELDS_DATALAB = [
        ('translation', FileLoaderField.SOURCE_LANGUAGE),
        ('translation', FileLoaderField.TARGET_LANGUAGE),
    ]

    JSON_FIELDS = [
        ('translation', FileLoaderField.SOURCE_LANGUAGE),
        ('translation', FileLoaderField.TARGET_LANGUAGE),
    ]
