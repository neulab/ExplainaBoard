from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
    TextFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.qa_open_domain)
class QAOpenDomainLoader(Loader):
    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.text

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = ["question", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("question", target_field_names[0], str),
                    FileLoaderField("answers", target_field_names[1], list),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("question_column", target_field_names[0], str),
                    FileLoaderField("answers_column", target_field_names[1], list),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        field_name = "predicted_answer"
        return {
            FileType.text: TextFileLoader(field_name, str),
        }
