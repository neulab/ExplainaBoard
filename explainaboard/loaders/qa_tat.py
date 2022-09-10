from __future__ import annotations

from explainaboard import TaskType
from explainaboard.constants import FileType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.qa_tat)
class QATatLoader(Loader):
    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.text

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField("question_column", "question", str),
                    FileLoaderField("context_column", "context", list),
                    FileLoaderField("table_column", "table", list),
                    FileLoaderField("answer_column", "true_answer", list),
                    FileLoaderField("answer_type_column", "answer_type", str),
                    FileLoaderField("answer_scale_column", "answer_scale", str),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField("answer", "predicted_answer", list),
                    FileLoaderField("scale", "predicted_answer_scale", str),
                ]
            )
        }
