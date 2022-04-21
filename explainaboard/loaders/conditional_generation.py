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

    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.tsv

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_names = ["source", "reference"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, field_names[0], str),
                    FileLoaderField(1, field_names[1], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("source", field_names[0], str),
                    FileLoaderField("references", field_names[1], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("source", field_names[0], str),
                    FileLoaderField("references", field_names[1], str),
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
    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_names = ["source", "reference"]
        return {
            FileType.tsv: TSVFileLoader(
                [
                    FileLoaderField(0, field_names[0], str),
                    FileLoaderField(1, field_names[1], str),
                ],
            ),
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("source", field_names[0], str),
                    FileLoaderField("references", field_names[1], str),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("text", field_names[0], str),
                    FileLoaderField("summary", field_names[1], str),
                ]
            ),
        }
