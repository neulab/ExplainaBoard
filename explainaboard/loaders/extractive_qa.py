from __future__ import annotations

from explainaboard.constants import FileType, TaskType
from explainaboard.loaders.file_loader import (
    DatalabFileLoader,
    FileLoader,
    FileLoaderField,
    JSONFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.loader_registry import register_loader


@register_loader(TaskType.qa_extractive)
class QAExtractiveLoader(Loader):
    @classmethod
    def default_dataset_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_output_file_type(cls) -> FileType:
        return FileType.json

    @classmethod
    def default_dataset_file_loaders(cls) -> dict[FileType, FileLoader]:
        target_field_names = ["context", "question", "answers"]
        return {
            FileType.json: JSONFileLoader(
                [
                    FileLoaderField(
                        target_field_names[0],
                        target_field_names[0],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField(
                        target_field_names[1],
                        target_field_names[1],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField(target_field_names[2], target_field_names[2]),
                ]
            ),
            FileType.datalab: DatalabFileLoader(
                [
                    FileLoaderField(
                        "context",
                        target_field_names[0],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField(
                        "question",
                        target_field_names[1],
                        str,
                        strip_before_parsing=False,
                    ),
                    FileLoaderField("answers", target_field_names[2]),
                ]
            ),
        }

    @classmethod
    def default_output_file_loaders(cls) -> dict[FileType, FileLoader]:
        return {
            FileType.json: JSONFileLoader(
                [FileLoaderField("predicted_answers", "predicted_answers")]
            )
        }
