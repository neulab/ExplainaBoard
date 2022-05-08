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
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label

    usage:
        please refer to `test_loaders.py`
    """

    OUTPUT_FIELDS = ['source', 'reference']
    JSON_FIELDS: list[str | tuple[str, str]] = ['source', 'reference']

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
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
                    FileLoaderField(cls.JSON_FIELDS[0], cls.OUTPUT_FIELDS[0], str),
                    FileLoaderField(cls.JSON_FIELDS[1], cls.OUTPUT_FIELDS[1], str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_name = "hypothesis"
        return {
            FileType.text: TextFileLoader(field_name, str),
            FileType.json: JSONFileLoader(
                [FileLoaderField(field_name, field_name, str)]
            ),
        }


@register_loader(TaskType.summarization)
class SummarizationLoader(ConditionalGenerationLoader):
    JSON_FIELDS = ['text', 'summary']


@register_loader(TaskType.machine_translation)
class MachineTranslationLoader(ConditionalGenerationLoader):
    JSON_FIELDS = [
        ('translation', FileLoaderField.SOURCE_LANGUAGE),
        ('translation', FileLoaderField.TARGET_LANGUAGE),
    ]
